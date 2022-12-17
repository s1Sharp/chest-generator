
import os
import zipfile
import zipfile
from controller.generator import create_chest_image_from_noise, Generator


GENERATED_PATH = 'generated'

NORMALIZE_NUMBER = 2**10
def normalize_seed(seed) -> int:
    return int(seed) % NORMALIZE_NUMBER


def zipdir(path, list_of_files, ziphandle):
    for file in list_of_files:
        ziphandle.write(file, 
                    os.path.relpath(file, os.path.join(path, '..')))


def generate_image_zip(app_path, model, rand_generator, image_seed: int, image_size: int, image_type: str = 'jpg') -> str:
    seed = image_seed

    """ Generate zip archive """
    zip_filepath = f"{GENERATED_PATH}/seed_{seed}_{image_size}_{image_type}.zip"
    full_zip_filepath = os.path.join(app_path, zip_filepath)


    if os.path.exists(full_zip_filepath):
        print('ret zip')
        return zip_filepath


    list_of_images = []
    for i in range(1, image_size + 1):
        img = generate_image(app_path, model, rand_generator, seed, i, image_type)
        list_of_images.append(os.path.join(app_path, img))


    with zipfile.ZipFile(full_zip_filepath, 'w', zipfile.ZIP_DEFLATED) as ziphandle:
        zipdir(f'{app_path}/{GENERATED_PATH}/', list_of_images, ziphandle)

    return zip_filepath



def generate_image(app_path, model, rand_generator, image_seed: int, image_number: int, image_type: str = 'jpg') -> str:
    seed = image_seed

    seed_dir = f"{app_path}/{GENERATED_PATH}/seed_{seed}"
    os.makedirs(seed_dir, exist_ok=True)

    """ Generate binary file and create zip archive """
    filepath = f"{GENERATED_PATH}/seed_{seed}/{image_number}.{image_type}"
    full_filepath = os.path.join(app_path, filepath)

    if os.path.exists(full_filepath):
        print('ret file')
        return filepath


    """ Create jpg file """
    jpg_data = create_chest_image_from_noise(model, rand_generator)
    new_file = open(full_filepath, 'wb')
    new_file.write(jpg_data)
    new_file.close()

    return filepath


if __name__ == '__main__':
    from generator import load_generator_from_file
    input_size = input ("Enter the desired image size: ")
    try:
        image_size = int(input_size)
    except ValueError:
        print("Not interesting value, need number(int)")
    if image_size > 0 and image_size < 1000:
        generate_image(load_generator_from_file("models/generator.pth"), image_size)
    else:
        print('Sorry, dont want to do this')

