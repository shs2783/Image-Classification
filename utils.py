def show_params(model):
    print("=" * 40, "Model Parameters", "=" * 40)
    num_params = 0
    for module_name, m in model.named_modules():
        if module_name == '':
            for name, params in m.named_parameters():
                print(name, params.size(), end=' ')
                i = 1
                for j in params.size():
                    i = i * j

                if 'weight' in name:
                    print('>'.rjust(20, '-'), i)
                else:
                    print()
                    
                num_params += i
    print('[*] Parameter Size: {}'.format(num_params))
    print("=" * 98)

def make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """

    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v