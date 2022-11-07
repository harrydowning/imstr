# imstr
<a href="https://pypi.org/project/imstr/">
  <img src="https://img.shields.io/pypi/v/imstr"/>
</a>

<ins>**Im**</ins>age <ins>**str**</ins>ing is a command line tool for converting images to strings.

## Install
The following command will install `imstr`
```
pip install imstr
```

## Usage
To use, simply call from the command line:
```
$ imstr --help

imstr

Usage:
  imstr [options] <image>

Options:
  --help                  Show this screen.
  -v --version            Show version.
  -o --output=FILENAME    Output target.
  -e --encoding=ENCODING  Output target encoding.
  -s --scale=SCALE        Scale output [default: 1].
  -w --width=WIDTH        Set width of output.
  -h --height=HEIGHT      Set height of output.
  -d --density=DENSITY    Set density string [default: .:-i|=+%O#@].
  -i --invert             Invert density string [default: False].
```

Alternatively it can be called as a python module:
```
$ python -m imstr --help
```
Or imported as a python library:
``` python
from imstr import imstr

output = imstr(...)
```

### Example
The following converts an input image (`cat.png`) to a string and saves it to a file (`cat.txt`).

Command line
```
$ imstr cat.png -o cat.txt
```
Python
``` python
from imstr import imstr

output = imstr('cat.png', filename='cat.txt')
```

### Example outputs

The following are made entirely out of textual characters

<div style="display: flex; justify-content: center; alight-items: center">
<img style="height: 250px" src="https://user-images.githubusercontent.com/74541141/200204655-4de7b4b5-7434-4a89-934b-e92c989cee0c.png" />
<img style="height: 250px" src="https://user-images.githubusercontent.com/74541141/200204656-731e4a81-a168-456f-bf2c-64ca34cec159.png" />
<img style="height: 250px" src="https://user-images.githubusercontent.com/74541141/200204663-765e1f32-0400-4510-82e1-923b02fd4d1f.png" />
</div>


