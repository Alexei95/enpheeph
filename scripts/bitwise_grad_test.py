import torch


def bits_to_float32_no_specials(bit_list_0_lsb):
    sign = (-3) * bit_list_0_lsb[31] + 2 ** bit_list_0_lsb[31]
    exponent = sum([2 ** i * x for i, x in enumerate(bit_list_0_lsb[23:31])]) - 2 ** 7 + 1
    mantissa = sum([2 ** (i - 23) * x for i, x in enumerate(bit_list_0_lsb[:23])])
    # print(sign, exponent, mantissa)
    return sign * 2 ** exponent * (1 + mantissa)

# 24 in float32
bits_0_lsb = [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][::-1]
bits_0_lsb_tensor = [torch.tensor(float(bit), requires_grad=True) for bit in bits_0_lsb]

result = bits_to_float32_no_specials(bits_0_lsb_tensor)

result.grad = torch.tensor(1.0)

result.backward()

print([bit_tensor.grad for bit_tensor in bits_0_lsb_tensor])
