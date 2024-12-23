def int_to_padded_binary(n, num_bin):
    return format(n, '0{num_bin}b'.format(num_bin = num_bin))[::-1]

def bin_to_state(n, num_int):
    state = []
    for j,i in enumerate(range(0,num_int,2)):
        match n[i:i+2]:
            case '10':
                state.append('\\nu_{{e}},p_{{{j}}}'.format(j=j))
            case '01':
                state.append('\\nu_{{\\mu}},p_{{{j}}}'.format(j=j))
            case '00':
                state.append('\\nu_{{e}},p_{{{j}}};\\nu_{{\\mu}},p_{{{j}}}'.format(j=j))
            case _:
                pass
        
    if(len(state) == 0):
        return '0'
    else:
        return ';'.join(state)
    