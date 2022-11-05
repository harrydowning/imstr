import numpy as np
import cv2 as cv
from docopt import docopt

_version = '0.0.1'
_density = '.:-i|=+%O#@'
_scale = 1
_invert = False
_cli = f"""imstr

Usage:
  imstr [options] <image> 

Options:
  --help                  Show this screen.
  -v --version            Show version.
  -o --output=FILENAME    Output target.
  -e --encoding=ENCODING  Output target encoding.
  -s --scale=SCALE        Scale output [default: {_scale}].
  -w --width=WIDTH        Set width of output.
  -h --height=HEIGHT      Set height of output.
  -d --density=DENSITY    Set density string [default: {_density}].
  -i --invert             Invert density string [default: {_invert}]. 
"""

def _density_mapping(density: str, normalised_intensity: float) -> str:
    index = int(np.round(normalised_intensity * (len(density) - 1)))
    return density[index]

def _get_imstr_array(image: np.ndarray, density: str) -> np.ndarray:
    info = np.iinfo(image.dtype)
    normalised_image = image / info.max
    vf = np.vectorize(lambda x: _density_mapping(density, x))
    return vf(normalised_image)

def _get_imstr(imstr_array: np.ndarray) -> str:
    imstr = ''
    for line in imstr_array:
        for char in line:
            imstr += char
        imstr += '\n'
    return imstr

def _write_imstr(imstr: str, filename: str, encoding: str):
    match filename:
        case None:
            print(imstr)
        case _:
            with open(filename, 'w', encoding=encoding) as file:
                file.write(imstr)

def _resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if width == None and height == None:
        return image
    
    im_height, im_width = image.shape

    if width == None:
        height = int(height)
        scale = height / im_height
        shape = (int(scale * im_width), height)
        return cv.resize(image, shape, interpolation=cv.INTER_AREA)
    
    if height == None:
        width = int(width)
        scale= width / im_width
        shape = (width, int(scale * im_height))
        return cv.resize(image, shape, interpolation=cv.INTER_AREA)

def _scale_image(image: np.ndarray, scale: float) -> np.ndarray:
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    return cv.resize(image, (width, height), interpolation=cv.INTER_AREA)

def imstr(image: str, /, filename: str = None, encoding: str = None, 
          scale: float = _scale, width: int = None, height: int = None,
          density: str = _density, invert: bool = _invert):
    """
    """

    image = cv.imread(image, cv.IMREAD_GRAYSCALE)
    resized_image = _resize_image(image, width, height)
    scaled_image = _scale_image(resized_image, scale)

    density = density[::-1] if invert else density

    imstr_array = _get_imstr_array(scaled_image, density)
    imstr = _get_imstr(imstr_array)
    
    _write_imstr(imstr, filename, encoding)

if __name__ == '__main__':
    args = docopt(_cli, version=f'imstr {_version}')
    imstr(args['<image>'], 
        filename=args['--output'],
        encoding=args['--encoding'], 
        scale=float(args['--scale']),
        width=args['--width'],
        height=args['--height'],
        density=args['--density'],
        invert=args['--invert'])
