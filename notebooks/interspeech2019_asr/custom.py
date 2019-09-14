import chainer
import torch
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy

class Reporter(chainer.Chain):
    def report(self, **kwargs):
        chainer.reporter.report(kwargs, self)

class ASRTransformer(ASRInterface, torch.nn.Module):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--label-smoothing", default=0.0, type=float)
        return parser

    def __init__(self, idim, odim, args=None):
        torch.nn.Module.__init__(self)
        self.encoder = Encoder(idim, input_layer="linear")
        self.decoder = Decoder(odim)
        self.criterion = LabelSmoothingLoss(odim, -1, args.label_smoothing, True)
        self.sos = odim - 1
        self.eos = odim - 1
        self.ignore_id=-1
        self.subsample = [0]
        self.reporter = Reporter()

    # for training
    def forward(self, xs_pad, ilens, ys_pad):
        """Compute scalar loss for backprop"""
        src_mask = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)

        ys_in_pad, ys_out_pad = self.add_sos_eos(ys_pad)
        ys_mask = self.target_mask(ys_in_pad)
        pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)

        loss = self.criterion(pred_pad, ys_out_pad)
        self.acc = th_accuracy(pred_pad.view(-1, pred_pad.size(-1)), ys_out_pad, ignore_label=self.ignore_id)
        self.reporter.report(loss=loss, acc=self.acc)
        return loss

    def add_sos_eos(self, ys_pad):
        from espnet.nets.pytorch_backend.nets_utils import pad_list
        eos = ys_pad.new([self.eos])
        sos = ys_pad.new([self.sos])
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        return pad_list(ys_in, self.eos), pad_list(ys_out, self.ignore_id)

    def target_mask(self, ys_in_pad):
        ys_mask = ys_in_pad != self.ignore_id
        m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
        return ys_mask.unsqueeze(-2) & m

    # for decoding
    def encode(self, feat):
        """Encode speech feature."""
        return self.encoder(feat.unsqueeze(0), None)[0][0]

    def scorers(self):
        """Scorer used in beam search"""
        return {"decoder": self.decoder}
