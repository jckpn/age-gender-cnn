# Label extraction functions

def binary_gender_label(filename):
    # Infer gender class from filename
    # e.g.  'F_30_1234.jpg' -> 0
    #       'M_30_1234.jpg' -> 1
    label = filename.split('_')[0].upper()
    label = (0 if label == 'F'
            else 1 if label == 'M'
            else None)
    return label

def age_int_label(filename, gender=None):
    # Infer age from filename
    # e.g.  'M_28_1234.jpg' -> 28
    #       'F_41_1234.jpg' -> 41
    if gender is not None and binary_gender_label(filename) is not gender:
        return None

    label = int(filename.split('_')[1])
    return label

def utkface_gcender_label(filename):
    # e.g.  '45_0_3_20170119171417728.jpg' -> 0 (male) -> 1 (to match other ds)
    label = int(filename.split('_')[1])
    label = (1 if label == 0 # gender labels are opposite of other datasets
            else 0 if label == 1
            else None)
    return label

def utkface_age_label(filename, gender=None):
    if gender is not None and utkface_gcender_label(filename) is not gender:
        return None
    
    label = int(filename.split('_')[0])
    return label

# def age_float_label(filename):
#     # Infer age from filename
#     # e.g.  'M_28_1234.jpg' -> 28
#     #       'F_41_1234.jpg' -> 41
#     label = float(filename.split('_')[1])
#     return label

# def age_class_label(cutoffs):
#     # Create unique class-getter function depending on the class_maxes
#     def f(filename):
#         # Get an age class from filename
#         # e.g.  'M_28_1234.jpg' -> 2
#         #       'F_41_1234.jpg' -> 4
#         label = age_int_label(filename)
#         for c, class_max in enumerate(cutoffs):
#             if label <= class_max:
#                 return c
#         return c+1
#     return f