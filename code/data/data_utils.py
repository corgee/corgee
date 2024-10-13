def parse_target(target):
    """ target types:
        - 'sum' -> also has 'terms' = list of dicts with {factor, lossname, lefttarget, righttarget}
        - 'dict' -> also has 'contents' = actual dict
    """
    if isinstance(target, str):
        # suitable for non-dict loss functions, format example: offer->text+0.5*offerimage->text
        # here offer, text and offerimage are encoder names
        # (factor, Optional[lossname], leftfeat, rightfeat)
        terms = []
        for lossterm in target.split('+'):
            factor = 1.0
            lossname = None
            if '*' in lossterm:
                factor, lossterm = lossterm.split('*')
                factor = float(factor)
            if ':' in lossterm:
                lossname, lossterm = lossterm.split(':')
            lefttarget, righttarget = lossterm.split('->')
            terms.append({
                'factor': factor,
                'lossname': lossname,
                'lefttarget': lefttarget,
                'righttarget': righttarget
            })
        res = {'type': 'sum', 'terms': terms}
    elif isinstance(target, dict):
        # suitable for dict loss functions, key is the name of evaluation metric, value is the indivdual target
        # format example: losstrainnce:offer->text+0.5*losstrainnce:offer->text
        # here losstrainnce is the name of loss to be used and individual target are passed in the same way as above
        res = {'type': 'dict'}
        res['contents'] = {}
        for key, value in target.items():
            res['contents'][key] = parse_target(value)
    else:
        raise NotImplementedError
    return res
