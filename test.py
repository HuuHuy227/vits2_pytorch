from text import _symbol_to_id, cleaned_text_to_sequence
# print(_symbol_to_id)
from text.vietnamese import g2p

p, t = g2p("Sau khi hù dọa, đối tượng còn yêu cầu chủ thuê bao đang nghe điện thoại chuyển tiền hoặc làm theo yêu cầu, hướng dẫn để phục vụ cho việc lừa đảo")
print(p)
print(t)
# print(cleaned_text_to_sequence(p, t))