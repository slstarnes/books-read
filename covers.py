from pathlib import Path
import requests
from PIL import Image, ImageOps
import pandas as pd
import math

fill_color = '#EAEAEA' # matches lstarnes.com background
override_images = {
    "Midnight in the Garden of Good and Evil": "https://upload.wikimedia.org/wikipedia/en/4/46/Midnight_in_the_Garden_of_Good_and_Evil_cover.jpg"
}
output_size = (180, 270)
modified_covers_output_folder = "modified_covers"

def download_file(url, output_path):
    r = requests.get(url, stream=True)
    with open(output_path, 'wb') as file:
        for chunk in r.iter_content(chunk_size=8192):
            file.write(chunk)
    return output_path

def download_cover(isbn, size="M", name=None, folder="covers"):
    url = f'https://covers.openlibrary.org/b/isbn/{isbn}-{size}.jpg'
    if name is None:
        name = isbn
    if folder:
        Path(folder).mkdir(exist_ok=True)
        file_path = Path(folder) / f"{name}-{size}.jpg"
    else:
        file_path = f"{name}-{size}.jpg"
    return download_file(url, file_path)


def download_images(book_titles, book_isbns):
    cover_files = []
    # books['Date Read'] = pd.to_datetime(books['Date Read'])
    for title, isbn in zip(book_titles, book_isbns):
    # for ix, row in books.query("`Exclusive Shelf` == 'read' & `Date Read`.dt.year == @YEAR_OF_ANALYSIS").sort_values('Date Read', ascending=True).iterrows():
        name = title.split(":")[0].split("(")[0].strip().replace("The", "").strip().lower().replace(" ", "-")
        if title in override_images:
            file = download_file(override_images[title], f"{name}.jpg")
        else:
            file = download_cover(isbn, name=name, size="M")
        if file is None:
            continue
        if Path(file).stat().st_size < 100:
            print(f"Error downloading cover for {name} - {isbn}")
            Path(file).unlink()
        else:
            cover_files.append(str(file))
    return cover_files

def resize_image(cover_path):
    output_size = (180, 270)
    Path(modified_covers_output_folder).mkdir(exist_ok=True)
    img = Image.open(cover_path)
    output_path = modified_covers_output_folder + "/" + Path(cover_path).name 
    w, h = img.size
    ratio = w / h
    if ratio > 0.85:
        ImageOps.fit(img, output_size).save(output_path)
    else:
        if h <= 270:
            ImageOps.pad(img, output_size, color=fill_color).save(output_path)
        else:
            ImageOps.fit(img, output_size).save(output_path)
    return output_path

def merge_images(files, horizontal=True):
    """Merge list images into single row of images, displayed side by side
    :param list of files: path first image file
    :return: the merged Image object
    """
    background_color = (255,255,255,0)
    img = []
    w = []
    h = []
    for f in files:
        if not isinstance(f, Image.Image):
            image = Image.open(f)
        else:
            image = f
        img.append(image)
        (width, height) = image.size
        w.append(width)
        h.append(height)

    if horizontal:
        result_width = sum(w)
        result_height = min(h)
    else:
        result_width = max(w)
        result_height = sum(h)
    
    result = Image.new('RGB', (result_width, result_height), color=background_color)
    for ix, image in enumerate(img):
        if horizontal:
            width = sum(w[:ix])
            result.paste(im=image, box=(width, 0))
        else:
            height = sum(h[:ix])
            result.paste(im=image, box=(0, height))
    return result


def create_composite(name: str, files_images: list, cols: int):
    rows = []
    for row in range(math.ceil(len(files_images) / cols)):
        fls = files_images[cols * row:cols * row + cols]
        if len(fls) > 0:
            rows.append(merge_images(fls, True))
    out = merge_images(rows, False)
    out.save(name)
    return out


def create_composite_images(image_files, variant_name="", column_set=(4,5,6,7,8,9)):
    fill_color = '#EAEAEA' # matches lstarnes.com background
    for c in column_set:
        create_composite(f'composite-{c}c{variant_name}.png', image_files, c)
        blanks_needed = c * math.ceil(len(image_files) / c) - len(image_files)
        [f.unlink() for f in Path(modified_covers_output_folder).glob("*_blank.jpg")]
        for ix in range(len(image_files), len(image_files) + blanks_needed):
            Image.new('RGB', output_size, color=(fill_color)).save(Path(modified_covers_output_folder) / f'cover_{ix:02d}_blank.jpg')
        image_files_plus_filled = image_files + list(Path(modified_covers_output_folder).glob("*_blank.jpg"))
        create_composite(f'composite-filled-{c}c{variant_name}.png', image_files_plus_filled, c)
        [f.unlink() for f in Path(modified_covers_output_folder).glob("*_blank.jpg")]




# covers = download_images(book_titles, book_isbns)
# cover_files_modified = [resize_image(c) for c in covers]
# create_composite_images(cover_files_modified)
# # remove a book (or 2) if you want to get a grid without missing spots....
# cover_files_modified_reduced = cover_files_modified.copy()
# cover_files_modified_reduced.remove("modified_covers/white-darkness-M.jpg")
# create_composite_images(cover_files_modified_reduced, "-36")