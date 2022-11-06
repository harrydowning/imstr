# imstr
A command line tool for converting an image to a string representation.

## Install

### Python Module

### Command Line Tool

## Usage
Convert an image to a string representation.

param image  
Filename for input image.

param filename  
Filename used to write output to a file.

param encoding  
Output target encoding.

param scale  
Scale the output. This will happen after any width or height resizing.

param width  
Set the width of the output. The height will be scaled accordingly if
not explicitly set.

param height  
Set the height of the output. The width will be scaled accordingly if
not explicitly set.

param density  
The string of characters used to replace pixel values.

param invert  
Invert the density string.

returns  
String representation of the input image.

raises ValueError  
Scale must be a non-negative, non-zero float.

raises ValueError  
Width must be a positive integer.

raises ValueError  
Height must be a positive integer.

### Examples
