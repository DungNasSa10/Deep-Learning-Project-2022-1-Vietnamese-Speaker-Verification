import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))
import re

gg_url = 'https://drive.google.com/file/d/1BL4tkLkPPDeuYwlCaMvIrKYCst8E4zEN/view?usp=share_link'

from src.utils.tools import get_voices_and_urls

# print(get_voices_and_urls(gg_url))

s = "view?usp=share_link"
print('#' in '\/%^&$?*#')
