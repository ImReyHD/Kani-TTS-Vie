import torch
from nemo.collections.tts.models import AudioCodecModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from nemo.utils.nemo_logging import Logger
from dataclasses import dataclass
import soundfile as sf
import os

nemo_logger = Logger()
nemo_logger.remove_stream_handlers()

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    from huggingface_hub import login

    login(token=HF_TOKEN)
else:
    print("HF_TOKEN not set; assuming HuggingFace credentials already cached.")

print(f"CUDA available: {torch.cuda.is_available()}")

@dataclass
class Config:
    model_name:str = 'pnnbao-ump/kani-tts-370m-vi'
    device_map:str = "auto"
    tokeniser_length:int = 64400
    start_of_text:int = 1
    end_of_text:int = 2
    max_new_tokens:int = 1200
    temperature:float = 0.2
    top_p:float = 0.95
    repetition_penalty:float = 1.1
    do_sample:bool = True

class NemoAudioPlayer:
    def __init__(self, config: Config, text_tokenizer_name: str = None):
        self.config = config
        self.nemo_codec_model = AudioCodecModel.from_pretrained("nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps").eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.nemo_codec_model.to(self.device)
        self.text_tokenizer_name = text_tokenizer_name
        if self.text_tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_tokenizer_name)
        self.tokeniser_length = self.config.tokeniser_length
        self.start_of_text = self.config.start_of_text
        self.end_of_text = self.config.end_of_text
        self.start_of_speech = self.tokeniser_length + 1
        self.end_of_speech = self.tokeniser_length + 2
        self.start_of_human = self.tokeniser_length + 3
        self.end_of_human = self.tokeniser_length + 4
        self.start_of_ai = self.tokeniser_length + 5
        self.end_of_ai = self.tokeniser_length + 6
        self.pad_token = self.tokeniser_length + 7
        self.audio_tokens_start = self.tokeniser_length + 10
        self.codebook_size = 4032

    def output_validation(self, out_ids):
        start_of_speech_flag = self.start_of_speech in out_ids
        end_of_speech_flag = self.end_of_speech in out_ids
        if not (start_of_speech_flag and end_of_speech_flag):
            raise ValueError('Special speech tokens not exist!')
    
    def get_nano_codes(self, out_ids):
        start_a_idx = (out_ids == self.start_of_speech).nonzero(as_tuple=True)[0].item()
        end_a_idx = (out_ids == self.end_of_speech).nonzero(as_tuple=True)[0].item()
        if start_a_idx >= end_a_idx:
            raise ValueError('Invalid audio codes sequence!')
        audio_codes = out_ids[start_a_idx+1 : end_a_idx]
        if len(audio_codes) % 4:
            raise ValueError('The length of the sequence must be a multiple of 4!')
        audio_codes = audio_codes.reshape(-1, 4)
        audio_codes = audio_codes - torch.tensor([self.codebook_size * i for i in range(4)])
        audio_codes = audio_codes - self.audio_tokens_start
        if (audio_codes < 0).sum().item() > 0:
            raise ValueError('Invalid audio tokens!')
        audio_codes = audio_codes.T.unsqueeze(0)
        len_ = torch.tensor([audio_codes.shape[-1]])
        return audio_codes, len_
    
    def get_text(self, out_ids):
        start_t_idx = (out_ids == self.start_of_text).nonzero(as_tuple=True)[0].item()
        end_t_idx = (out_ids == self.end_of_text).nonzero(as_tuple=True)[0].item()
        txt_tokens = out_ids[start_t_idx : end_t_idx+1]
        text = self.tokenizer.decode(txt_tokens, skip_special_tokens=True)
        return text
    
    def get_waveform(self, out_ids):
        out_ids = out_ids.flatten()
        self.output_validation(out_ids)
        audio_codes, len_ = self.get_nano_codes(out_ids)
        audio_codes, len_ = audio_codes.to(self.device), len_.to(self.device)
        with torch.inference_mode():
            reconstructed_audio, _ = self.nemo_codec_model.decode(tokens=audio_codes, tokens_len=len_)
            output_audio = reconstructed_audio.cpu().detach().numpy().squeeze()
        if self.text_tokenizer_name:
            text = self.get_text(out_ids)
            return output_audio, text
        else:
            return output_audio, None

class KaniModel:
    def __init__(self, config: Config, player: NemoAudioPlayer):
        self.config = config
        self.player = player
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name, torch_dtype=torch.bfloat16, device_map=self.config.device_map)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

    def get_input_ids(self, text_prompt: str, speaker_id: str = None):
        start_of_human = self.player.start_of_human
        end_of_text = self.player.end_of_text
        end_of_human = self.player.end_of_human
        if speaker_id is not None:
            text_prompt = f"{speaker_id.lower()}: {text_prompt}"
        text_ids = self.tokenizer.encode(text_prompt, add_special_tokens=True)
        text_ids.append(end_of_text)
        text_ids_tensor = torch.tensor([text_ids], dtype=torch.int64)
        start_token = torch.tensor([[start_of_human]], dtype=torch.int64)
        end_token = torch.tensor([[end_of_human]], dtype=torch.int64)
        modified_input_ids = torch.cat([start_token, text_ids_tensor, end_token], dim=1)

        attention_mask = torch.ones(1, modified_input_ids.shape[1], dtype=torch.int64)
        return modified_input_ids, attention_mask

    def model_request(self, input_ids: torch.tensor, attention_mask: torch.tensor):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                num_return_sequences=1,
                eos_token_id=self.player.end_of_speech,
            )
        return generated_ids.to('cpu')

    def run_model(self, text: str, speaker_id: str = None):
        input_ids, attention_mask = self.get_input_ids(text, speaker_id)
        model_output = self.model_request(input_ids, attention_mask)
        audio, _ = self.player.get_waveform(model_output)
        return audio, text

config = Config()
player = NemoAudioPlayer(config)
kani = KaniModel(config, player)
prompt = "Hoặc bạn ra ngoài tìm kiếm những người có khả năng vô địch và mang họ về đội ngũ của mình."
audio, text = kani.run_model(prompt, speaker_id="nu-mien-bac")
print(text)
sf.write("audio-nu-mien-bac.wav", audio, samplerate=22050)
print("Audio saved to audio-nu-mien-bac.wav")
audio, text = kani.run_model(prompt, speaker_id="nu-mien-nam")
print(text)
sf.write("audio-nu-mien-nam.wav", audio, samplerate=22050)
print("Audio saved to audio-nu-mien-nam.wav")
audio, text = kani.run_model(prompt, speaker_id="nam-mien-bac")
print(text)
sf.write("audio-nam-mien-bac.wav", audio, samplerate=22050)
print("Audio saved to audio-nam-mien-bac.wav")
audio, text = kani.run_model(prompt, speaker_id="nam-mien-nam")
print(text)
sf.write("audio-nam-mien-nam.wav", audio, samplerate=22050)
print("Audio saved to audio-nam-mien-nam.wav")