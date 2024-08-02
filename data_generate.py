import random

vocab_list = ["[BOS]", "[EOS]", "[PAD]", 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
bos_token = "[BOS]"
eos_token = "[EOS]"
pad_token = "[PAD]"

#加密规则，每个字符，ASCII循环减5，然后逆序，如：abcfg->bxwv
source_path = r"D:\OneDrive\桌面\transformer\source.txt"
target_path = r"D:\OneDrive\桌面\transformer\target.txt"
with open(source_path, 'w') as f:
    pass
with open(target_path, 'w') as f:
    pass

for _ in range(10000):
    source_str = ""
    target_str = ""
    for idx in range(random.randint(3, 10)):
        i = random.randint(0,25)
        source_str += char_list[i]
        target_str += char_list[(i + 26 - 5) % 26]
    target_str = target_str[::-1]
    with open(source_path, 'a') as f:
        f.write(source_str + '\n')
    with open(target_path, 'a') as f:
        f.write(target_str + '\n')
