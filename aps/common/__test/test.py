
from aps.common.logger import _InnerColors, _strip_colored_string, _get_colored_string, LoggerColor, LoggerHighlights, LoggerAttributes

c = '\033[1m\033[100m\033[31mcacca\033[00m'

print(c[1:4] == '[1m')
print(c[1:4] == _InnerColors.BOLD[1:])
print(c[-4:] == '[00m')
print(c[-4:] == _InnerColors.END[1:])


print(c)
print(c[4:])

x = [x for x in c[1:4]]
print(x)

x = [x for x in c[-4:]]
print(x)

text, color, highlight, attrs = _strip_colored_string(c)
print(_strip_colored_string(c))

print(_get_colored_string(text, color, highlight, attrs))
