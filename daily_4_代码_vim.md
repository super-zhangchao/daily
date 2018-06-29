Vim 的 End/Home 键无效的解决方法
    1    E  VK_END                "/033[4~"
    2    E  VK_HOME               "/033[1~"
    3    E  VK_INSERT             "/033[2~"
    4    E  VK_DELETE             "/033[3~"

delete   删除左边字符
x        删除当前光标下的字符 没有删除右边字符
dw       删除光标之后的单词剩余部分。
d$       删除光标之后的该行剩余部分。
dd       删除当前行。
c        功能和d相同，区别在于完成删除操作后进入INSERT MODE
cc       也是删除当前行，然后进入INSERT MODE
	
	
	
	
	