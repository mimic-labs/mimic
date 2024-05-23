def import_markdown(path):
    with open(path, 'r', encoding="utf8") as f:
        markdown = f.read()
    return markdown