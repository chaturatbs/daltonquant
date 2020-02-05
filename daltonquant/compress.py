# Wrap DaltonQuant's compression in a function

from .compressors import PngQuant, TinyPng, MedianCutQuant, SpecimenQuant
from .scorers import LinearTransform, NonLinearTransform

def compress(impath, userid, database_path, transform_name='linear', prequantizer_name='pngquant', **kwargs):
    if transform_name == 'linear':
        transform = LinearTransform
    elif transform_name == 'non-linear':
        transform = NonLinearTransform
    else:
        raise ValueError("Unknown transform name %s" % transform_name)
    
    if prequantizer_name == 'pngquant':
        prequantizer = PngQuant
    elif prequantizer_name == 'tinypng':
        prequantizer = TinyPng
    elif prequantizer_name == 'mediancut':
        prequantizer = MedianCutQuant
    else:
        raise ValueError("Unknown pre-quantizer name %s" % prequantizer_name)
    
    transformer = transform(userid, database_path=database_path)
    specimen = SpecimenQuant(scorer, quantizer=prequantizer())
    return specimen.compress(impath, ncolors=target_ncolors, ret_im=True)