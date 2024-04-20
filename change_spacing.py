import SimpleITK as sitk

image_path = r"C:\Users\Sofia Sappia\Desktop\87c4e690-b5b2-43a5-8ca1-1bbdfccf44b3.mha"
# load the image
image = sitk.ReadImage(image_path)

# print the spacing
print(image.GetSpacing())

# change the spacing to 0.28mm in all directions
image.SetSpacing([0.28, 0.28, 0.28])
