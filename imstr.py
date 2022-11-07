import sys, os
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

_scale_err_msg = 'Scale must be a non-negative, non-zero float'
_width_err_msg = 'Width must be a positive integer'
_height_err_msg = 'Height must be a positive integer'

_is_using_cli = False

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

def _write_imstr(imstr: str, filename: str | None, encoding: str | None):
    if filename == None:
        if encoding != None:
            sys.stdout.buffer.write(imstr.encode(encoding))
        else:
            sys.stdout.write(imstr)
    else:
        with open(filename, 'w', encoding=encoding) as file:
                    file.write(imstr)

def _resize_image(image: np.ndarray, width: int | None, 
                  height: int | None) -> np.ndarray:
    if width == None and height == None:
        return image
    
    im_height, im_width = image.shape

    if width == None:
        scale = height / im_height
        new_width = max(int(scale * im_width), 1)
        shape = (new_width, height)
        return cv.resize(image, shape, interpolation=cv.INTER_AREA)
    
    if height == None:
        scale = width / im_width
        new_height = max(int(scale * im_height), 1)
        shape = (width, new_height)
        return cv.resize(image, shape, interpolation=cv.INTER_AREA)

def _scale_image(image: np.ndarray, scale: float) -> np.ndarray:
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    return cv.resize(image, (width, height), interpolation=cv.INTER_AREA)

def _resolve_error(error, err_msg):
    if _is_using_cli:
        print(err_msg, file=sys.stderr)
        exit(1)
    else:
        if error == None:
            raise # Propogate the error
        raise error(err_msg)

def _handle_value_error(value, value_fn, predicate, err_msg, default = None):
    try:
        value = value_fn(value)
        if predicate(value):
            raise ValueError(err_msg)
    except ValueError:
        _resolve_error(None, err_msg)
    except TypeError:
        return default
    return value

def imstr(image: str, filename: str | None = None, 
          encoding: str | None = None, scale: float = _scale, 
          width: int | None = None, height: int | None = None,
          density: str = _density, invert: bool = _invert):
    """
    Convert an image to a string representation.

    :param image: Filename for input image.
    :param filename: Filename used to write output to a file.
    :param encoding: Output target encoding.
    :param scale: Scale the output. This will happen after any width or height
    resizing.
    :param width: Set the width of the output. The height will be scaled
    accordingly if not explicitly set.
    :param height: Set the height of the output. The width will be scaled
    accordingly if not explicitly set.
    :param density: The string of characters used to replace pixel values.
    :param invert: Invert the density string.
    
    :returns: String representation of the input image.

    :raises ValueError: Scale must be a non-negative, non-zero float
    :raises ValueError: Width must be a positive integer
    :raises ValueError: Height must be a positive integer
    :raises FileNotFoundError: Image must be a valid path to a file that exists
    :raises LookupError: Encoding must be valid
    """
    scale = _handle_value_error(scale, float, lambda x: x <= 0,
                                _scale_err_msg, _scale)
    width = _handle_value_error(width, int, lambda x: x < 1, _width_err_msg)
    height = _handle_value_error(height, int, lambda x: x < 1, _height_err_msg)

    if not os.path.isfile(image):
        fnf_err_msg = f"The file '{image}' could not be found."
        _resolve_error(FileNotFoundError, fnf_err_msg)

    image_array = cv.imread(image, cv.IMREAD_GRAYSCALE)
    resized_image = _resize_image(image_array, width, height)
    scaled_image = _scale_image(resized_image, scale)
    density = density[::-1] if invert else density

    imstr_array = _get_imstr_array(scaled_image, density)
    imstr = _get_imstr(imstr_array)
    
    if _is_using_cli or filename != None:
        try:
            _write_imstr(imstr, filename, encoding)
        except LookupError:
            _resolve_error(None, 'Could not find specified encoding.')
    
    return imstr

def main():
    global _is_using_cli 
    _is_using_cli = True

    args = docopt(_cli, version=f'imstr {_version}')
    imstr(image=args['<image>'], filename=args['--output'],
          encoding=args['--encoding'], scale=args['--scale'],
          width=args['--width'], height=args['--height'],
          density=args['--density'], invert=args['--invert'])

if __name__ == '__main__':
    main()
