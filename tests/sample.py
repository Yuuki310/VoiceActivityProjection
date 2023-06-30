
event = {'pred_backchannel': [[], [], [(180, 205, 1)], [(383, 408, 1), (477, 502, 1)]], 'pred_backchannel_neg': [[], [], [], []], 'shift': [[], [], [], []], 'hold': [[(254, 303, 0)], [], [], []], 'long': [[], [], [], []], 'pred_shift': [[], [], [], []], 'pred_shift_neg': [[(229, 254, 0)], [], [], []], 'short': [[], [], [(205, 219, 1)], [(408, 425, 1), (502, 527, 1)]]}


def extract_prediction_and_targets(
    self,
    p_now: Tensor,
    p_fut: Tensor,
    events: Dict[str, List[List[Tuple[int, int, int]]]],
    device=None,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    batch_size = len(events["hold"])
    preds = {"hs": [], "pred_shift": [], "ls": [], "pred_backchannel": []}
    targets = {"hs": [], "pred_shift": [], "ls": [], "pred_backchannel": []}

    for b in range(batch_size):
        ###########################################
        # Hold vs Shift
        ###########################################
        # The metrics (i.e. shift/hold) are binary so we must decide
        # which 'class' corresponds to which numeric label
        # we use Holds=0, Shifts=1
        for start, end, speaker in events["shift"][b]:
            pshift = p_now[b, start:end, speaker]
            preds["hs"].append(pshift)
            targets["hs"].append(torch.ones_like(pshift))

        for start, end, speaker in events["hold"][b]:
            phold = 1 - p_now[b, start:end, speaker]
            preds["hs"].append(phold)
            targets["hs"].append(torch.zeros_like(phold))
        ###########################################
        # Shift-prediction
        ###########################################
        for start, end, speaker in events["pred_shift"][b]:
            # prob of next speaker -> the correct next speaker i.e. a SHIFT
            pshift = p_fut[b, start:end, speaker]
            preds["pred_shift"].append(pshift)
            targets["pred_shift"].append(torch.ones_like(pshift))
        for start, end, speaker in events["pred_shift_neg"][b]:
            # prob of next speaker -> the correct next speaker i.e. a HOLD
            phold = 1 - p_fut[b, start:end, speaker]  # 1-shift = Hold
            preds["pred_shift"].append(phold)
            # Negatives are zero -> hold predictions
            targets["pred_shift"].append(torch.zeros_like(phold))
        ###########################################
        # Backchannel-prediction
        ###########################################
        # TODO: Backchannel with p_now/p_fut???
        # for start, end, speaker in events["pred_backchannel"][b]:
        #     # prob of next speaker -> the correct next backchanneler i.e. a Backchannel
        #     pred_bc = p_bc[b, start:end, speaker]
        #     preds["pred_backchannel"].append(pred_bc)
        #     targets["pred_backchannel"].append(torch.ones_like(pred_bc))
        # for start, end, speaker in events["pred_backchannel_neg"][b]:
        #     # prob of 'speaker' making a 'backchannel' in the close future
        #     # over these negatives this probability should be low -> 0
        #     # so no change of probability have to be made (only the labels are now zero)
        #     pred_bc = p_bc[b, start:end, speaker]  # 1-shift = Hold
        #     preds["pred_backchannel"].append(
        #         pred_bc
        #     )  # Negatives are zero -> hold predictions
        #     targets["pred_backchannel"].append(torch.zeros_like(pred_bc))
        ###########################################
        # Long vs Shoft
        ###########################################
        # TODO: Should this be the same as backchannel
        # or simply next speaker probs?
        for start, end, speaker in events["long"][b]:
            # prob of next speaker -> the correct next speaker i.e. a LONG
            plong = p_fut[b, start:end, speaker]
            preds["ls"].append(plong)
            targets["ls"].append(torch.ones_like(plong))
        for start, end, speaker in events["short"][b]:
            # the speaker in the 'short' events is the speaker who
            # utters a short utterance: p[b, start:end, speaker] means:
            # the  speaker saying something short has this probability
            # of continue as a 'long'
            # Therefore to correctly predict a 'short' entry this probability
            # should be low -> 0
            # thus we do not have to subtract the prob from 1 (only the labels are now zero)
            # prob of next speaker -> the correct next speaker i.e. a SHORT
            pshort = p_fut[b, start:end, speaker]  # 1-shift = Hold
            preds["ls"].append(pshort)
            # Negatives are zero -> short predictions
            targets["ls"].append(torch.zeros_like(pshort))

    # cat/stack/flatten to single tensor
    device = device if device is not None else p_now.device
    out_preds = {}
    out_targets = {}
    print("objective")
    print(preds)
    for k, v in preds.items():
        if len(v) > 0:
            out_preds[k] = torch.cat(v).to(device)
        else:
            out_preds[k] = None
    for k, v in targets.items():
        if len(v) > 0:
            out_targets[k] = torch.cat(v).long().to(device)
        else:
            out_targets[k] = None
    return out_preds, out_targets

if __name__ == "__main__":
    extract_prediction_and_targets(event)
    print(ob)
