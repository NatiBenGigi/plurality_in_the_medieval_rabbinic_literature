def clean_text2(originalText, encode_value="utf8", remove_header=False):
    if originalText in ["", None]:
        return "", ""
    
    keep_first_break_line = True
    title = extractTitle(originalText)
    clean_content = ""
    idx = 0
    originalTextLen = len(originalText)
    prev_c_unicode = 0

    for c in originalText:
        c_unicode = ord(c)
        
        if c_unicode in (list(range(48, 58)) +  # Numbers
                         list(range(65, 91)) +  # Uppercase Latin alphabet
                         list(range(97, 123)) +  # Lowercase Latin alphabet
                         [47, 43, 42, 824, 92, 6, 7, 12, 225, 64, 160]):  # Specific symbols
            pass
        elif c_unicode == 177:  # ± to "
            clean_content += '״'
        elif c_unicode == 8801:  # ≡ to ´
            clean_content += "´"
        elif c_unicode in [41, 40]:  # Parentheses
            if encode_value == "cp1255":
                clean_content += " {" if c_unicode == 41 else " }"
            else:
                clean_content += " }" if c_unicode == 41 else " {"
        elif c_unicode in [123, 125, 91, 93, 58, 59, 46, 44, 63, 33, 34, 39]:
            # Punctuation
            clean_content += f" {c} "
        elif c_unicode == 61:  # Equal sign
            clean_content += f" {c} " if prev_c_unicode in [32, 0] else f"{c} "
        elif c_unicode == 10:  # New line
            if keep_first_break_line:
                clean_content += c
                keep_first_break_line = False
            elif idx != originalTextLen:
                clean_content += " "
        else:
            if 1488 <= c_unicode <= 1514:  # Hebrew letters
                if idx < 2 or (originalTextLen - idx < 2):
                    clean_content += c
                else:
                    if ((originalText[idx-1] == "1" and ord(originalText[idx-2]) == 6) or
                        (originalText[idx-2] == "1" and ord(originalText[idx-3]) == 6)) and \
                       (ord(originalText[idx+1]) == 7 or ord(originalText[idx+2]) == 7):
                        pass
                    else:
                        clean_content += c
            else:
                clean_content += c

        idx += 1
        prev_c_unicode = ord(c)

    if remove_header:
        clean_content = remove_text_header(clean_content)

    clean_content = ' '.join(clean_content.split())
    return clean_content, title

def remove_text_header(originalText):
    clean_text = ""
    header_removed = False
    char_index = 0
    
    for c in originalText:
        char_index += 1
        if ord(c) == 10 and not header_removed:
            clean_text = ""
            header_removed = True
        else:
            clean_text += c
        if char_index >= 50:
            header_removed = True

    return clean_text

def extractTitle(Text):
    if Text in ["", None]:
        return ""
    
    title = ""
    for c in Text:
        if ord(c) == 10 or c == ".":
            break
        title += c
    
    return title
