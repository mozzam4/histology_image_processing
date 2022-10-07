import os
import json
from colorthief import ColorThief
# The patch directory must contain individual folders of each image containing patches of that image
patch_dir = r'C:\Users\mozza\Documents\test\patches'


def create_json(input_path):
   pallette_dict_list = []
   for filename in os.listdir(input_path):
      print(filename)
      pallette_dict = {}
      input_filename = os.path.join(input_path, filename)
      img = ColorThief(input_filename)
      pallette = img.get_palette(color_count=10)
      # pallette = np.array(pallette)
      pallette_dict['Filename'] = input_filename
      pallette_dict['pallette'] = pallette
      pallette_dict_list.append(pallette_dict)

   if len(pallette_dict_list) > 0:
      output_path = input_path + '.json'
      with open(output_path, 'w') as final:
         json.dump(pallette_dict_list, final)


if __name__ == '__main__':
   all_input_dirs = []
   for filename in os.listdir(patch_dir):
      all_input_dirs.append(os.path.join(patch_dir, filename))
   for each in all_input_dirs:
      create_json(each)
