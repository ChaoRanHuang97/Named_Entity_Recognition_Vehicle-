import re
def format_phone_number(detected_number):
    detected_number = detected_number.lower().split('\n')[0]
    char_to_num = {
        '!': '1',
        '@': '2',
        '#': '3',
        '$': '4',
        '%': '5',
        '^': '6',
        '&': '7'
    }
    help_dict = {
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'zero': '0',
        'too': '2',
        'ten': '10',
        'twenty': '20',
        'thirty': '30',
        'forty': '40',
        'fifty': '50',
        'sixty': '60',
        'seventy': '70',
        'eighty': '80',
        'ninety': '90'
    }
    converted_number = ''.join(char_to_num.get(ele, ele) for ele in detected_number)
    converted_number = re.sub(r'[^a-zA-Z0-9]', ' ', converted_number)
    for digit in help_dict.keys():
        if digit in converted_number:
            converted_number = converted_number.replace(digit, help_dict[digit])
    converted_number = ''.join(help_dict.get(ele, ele) for ele in converted_number.split())
    converted_number = converted_number.replace('o', '0').replace('i', '1').replace('l', '1')
    converted_number = re.sub("\D", "", converted_number)
    if len(converted_number) == 10 and converted_number[0:10].isdigit():
        return converted_number[0:10]
    else:
        return None