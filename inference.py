## LJSpeech
import torch

import utils
from models import SynthesizerTrn
from text.symbols import symbols

from scipy.io.wavfile import write
from text.cleaners import clean_text
from text.vietnamese import g2p
from text import cleaned_text_to_sequence

def get_text(text):
    phone, tone = clean_text(text)
    phone, tone = cleaned_text_to_sequence(phone, tone)
    # print(len(phone), len(tone))
    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    return phone,  tone

CONFIG_PATH = "./configs/config_test.json"
MODEL_PATH = "D:/demo\checkpoints spd/G_71000.pth"   
OUTPUT_WAV_PATH = "sample2.wav"

# CONFIG_PATH = "./configs/config_nosdp.json"
# MODEL_PATH = "D:\demo\checkpoints\G_70000.pth"
# OUTPUT_WAV_PATH = "sample1.wav"

TEXT = """
Theo Công an tỉnh Hải Dương, cùng với việc bắt giữ được Vũ Thị Tuyết, phòng Cảnh sát hình sự Công an tỉnh đã kịp thời ngăn chặn hành vi câu kết với các đối tượng khác của Tuyết để xuất cảnh sang thành phố Bamako, nước Cộng hòa Mali Châu Phi vào ngày ba để hành nghề mại dâm nhằm trốn tránh sự phát hiện của cơ quan chức năng, điều hành nhóm Phố đèn đỏ Hải Dương
"""



hps = utils.get_hparams_from_file(CONFIG_PATH)

if (
    "use_mel_posterior_encoder" in hps.model.keys()
    and hps.model.use_mel_posterior_encoder == True
):
    print("Using mel posterior encoder for VITS2")
    posterior_channels = 80  # vits2
    hps.data.use_mel_posterior_encoder = True
else:
    print("Using lin posterior encoder for VITS1")
    posterior_channels = hps.data.filter_length // 2 + 1
    hps.data.use_mel_posterior_encoder = False

net_g = SynthesizerTrn(
    len(symbols),
    posterior_channels,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model
).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint(MODEL_PATH, net_g, None, skip_optimizer=True)

stn_tst, tone = get_text(TEXT)
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    tone = tone.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, tone, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
#ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))

write(data=audio, rate=hps.data.sampling_rate, filename=OUTPUT_WAV_PATH)
