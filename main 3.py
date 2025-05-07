import os
import torch
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import numpy as np

import sounddevice as sd
import soundfile as sf

duration = 5 # seconds
sample_rate = 16000

print("Recording...")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
sd.wait()  # Wait until recording is finished
sf.write("sample.wav", audio, sample_rate)
print("Saved recording to sample.wav")

class SimpleSpeechToText:
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and processor
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        
        # Ignore the warning about masked_spec_embed - it's normal for fine-tuned models
        print("Model loaded successfully!")

    def transcribe(self, audio_path):
        try:
            # Load audio file (convert to mono if needed)
            audio_input, sample_rate = sf.read(audio_path)
            if len(audio_input.shape) > 1:
                audio_input = np.mean(audio_input, axis=1)
                
            # Process audio
            inputs = self.processor(
                audio_input, 
                sampling_rate=sample_rate, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            return self.processor.batch_decode(predicted_ids)[0]
            
        except Exception as e:
            return f"Error: {str(e)}"

def main():
    # Initialize the model
    stt = SimpleSpeechToText()
    
    # Test with an audio file (create a sample if needed)
    audio_file = "sample.wav"
    
    # Create a test audio file if it doesn't exist (1 second of silence)
    if not os.path.exists(audio_file):
        print(f"Creating test audio file '{audio_file}'...")
        sf.write(audio_file, np.zeros(16000), 16000)  # 1 second of silence at 16kHz
    
    # Transcribe
    result = stt.transcribe(audio_file)
    print("\nTranscription Result:")
    print(result)

if __name__ == "__main__":
    main()