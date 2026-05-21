import re

with open('output/article.md', encoding='utf-8') as f:
    text = f.read()

# Убираем H1 и строку-подпись в самом начале
text = re.sub(r'^# .+?\n\n\*.+?\*\n\n---\n\n', '', text, flags=re.DOTALL)

# Заменяем пути к картинкам
text = text.replace('../pd_multiplier/output/figures/', './figures/')

with open('output/article_blog.md', 'w', encoding='utf-8') as f:
    f.write(text)

print('Done, chars:', len(text))
print(text[:300])
