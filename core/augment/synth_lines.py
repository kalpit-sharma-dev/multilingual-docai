from PIL import Image, ImageDraw, ImageFont
import random, os, textwrap
import numpy as np

# Simple mixed-script line generator.
# Uses system fonts; adjust font paths if needed.

DEFAULT_FONTS = {
    'LATIN': '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    'DEV': '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf',
    'ARAB': '/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf'
}

EXAMPLES = {
    'LATIN': ['Invoice #12345 due 31/12/2024', 'Total: $1,250.00', 'Phone: +1-555-1234'],
    'DEV': ['कुल राशि ₹2,350.00', 'संपर्क: 01234 567890', 'धन्यवाद'],
    'ARAB': ['المبلغ الإجمالي ٢٣٥٠٫٠٠', 'هاتف: ٠١٢٣٤٥٦٧٨٩', 'شكراً']
}

def load_font(path, size):
    try:
        return ImageFont.truetype(path, size=size)
    except Exception:
        return ImageFont.load_default()

def render_mixed_line(width=1600, height=64, segments=2):
    img = Image.new('RGB', (width, height), (255,255,255))
    draw = ImageDraw.Draw(img)
    x = 10
    transcript = []
    tags = []
    for i in range(segments):
        script = random.choice(['LATIN','DEV','ARAB'])
        text = random.choice(EXAMPLES[script])
        font_path = DEFAULT_FONTS.get(script)
        font = load_font(font_path, size=random.randint(26,40))
        draw.text((x, max(2, height//4)), text, font=font, fill=(0,0,0))
        w, h = draw.textsize(text, font=font)
        x += w + random.randint(10,60)
        transcript.append(text)
        tags.append(script)
    return img, transcript, tags

def generate_mixed_lines_batch(batch_size=8, **kwargs):
    samples = []
    for _ in range(batch_size):
        img, transcript, tags = render_mixed_line(**kwargs)
        samples.append({"image": img, "transcript": transcript, "tags": tags})
    return samples

if __name__ == '__main__':
    img, transcript, tags = render_mixed_line()
    os.makedirs('debug', exist_ok=True)
    img.save('debug/synth_line.png')
    print('Saved debug/synth_line.png, transcript:', transcript, 'tags:', tags)
    # Example batch generation
    batch = generate_mixed_lines_batch(batch_size=4)
    for i, sample in enumerate(batch):
        sample["image"].save(f'debug/synth_line_{i}.png')
        print(f'Batch {i}: transcript={sample["transcript"]}, tags={sample["tags"]}')
