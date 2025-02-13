import json
from math import log10, floor
from hyperparam_search import transform

def find_exp(number) -> int:
    base10 = log10(abs(number))
    return int(floor(base10))


def decompose(number):
    exp = find_exp(number)
    base = (number / (10 ** exp))
    return base, exp


def get_ten(exp):
    return f'$10^{{{exp}}}$'

with open('search/search_results.json') as fp:
    lines = fp.read().split('\n')

    for i, line in enumerate(lines[:-1]):
        output = '    ' + str(i+1) + ' & '
        data = json.loads(line)
        par = data['params']
        slr_base, slr_exp = decompose(par['slr'])
        rlr_base, rlr_exp = decompose(par['rlr'])
        if par['lc_multiplier'] == 0:
            lc_cell = '\\multicolumn{2}{c}{0}'
        else:
            lc_base, lc_exp = decompose(par['length_cost'])
            lc_cell = f'{lc_base:.2f} & $10^{{{lc_exp}}}$'
        params = (
            f"{data['target']:.2f}",
            f'{slr_base:.1f}', 
            get_ten(slr_exp),
            f'{rlr_base:.1f}',
            get_ten(rlr_exp),
            str(int(par['vocab_size'])),
            str(int(par['hidden_units'])),
            lc_cell,
            par['mode'],
        )
        #params = transform(data['params'])
        #params = (
        #    f'{}
        #    f'\\lrate{{{params["slr"]:.5f}}}',
        #    f'\\lrate{{{params["slr"] * params["rlr_multiplier"]:.5f}}}',
        #   str(int(params['vocab_size'])),
        #    str(int(params['hidden_units'])),
        #    f'\\lrate{{{params["length_cost"] * params["lc_multiplier"]:.7f}}}' if params["lc_multiplier"] != 0 else '0',
        #)
        output += ' & '.join(params)
        output += ' \\\\'
        print(output)

print(len(params)+2, 'cells')

