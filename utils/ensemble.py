import torch

def forward_x8_tta(model, lr_img):
    """ 8次几何自集成推理 """
    def _transform(v, op):
        if op == 'v': return v.flip(2)
        elif op == 'h': return v.flip(3)
        elif op == 't': return v.transpose(2, 3)
        return v

    ops = ['', 'v', 'h', 't', 'v,t', 't,v', 'h,t', 't,h']
    sr_list = []
    
    for op in ops:
        lr_in = lr_img.clone()
        for trans in op.split(','):
            if trans: lr_in = _transform(lr_in, trans)
            
        with torch.no_grad():
            sr_out = model(lr_in)
            
        # 逆变换还原
        inv_op = op
        if op == 'v,t': inv_op = 't,v'
        elif op == 't,v': inv_op = 'v,t'
        elif op == 'h,t': inv_op = 't,h'
        elif op == 't,h': inv_op = 'h,t'
        
        for trans in inv_op.split(','):
            if trans: sr_out = _transform(sr_out, trans)
            
        sr_list.append(sr_out)
        
    return torch.mean(torch.stack(sr_list), dim=0).clamp(0, 1)
