import re

text =  "My phone number is +49 012345689. My brother's one is 0123465488"
result = re.findall('\+?\d+\s?\d+', text)
print(result)

email = "My email is nhviet1009@gmail.com. I have another email, which is vietnguyen@sporttotal.com"
print(re.findall('\w+\@\w+\.\w+', email))