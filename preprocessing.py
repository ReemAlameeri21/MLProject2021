import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import pydicom as dicom
import numpy as np
import PIL as Image
from skimage import io
from skimage import color
from skimage.io import imread


df= pd.read_csv('train.csv')
df.columns=["ID", "Labels"]

le = LabelEncoder()
le.fit(df["Labels"])
le.classes_
df["Labels"]=le.transform(df["Labels"])

#Atypical=0, Indeterminate= 1, Negative=2, Typical=3
for i in os.listdir('RSNACOVID/train/'):
     for j in os.listdir('RSNACOVID/train/'+i):
         for k in os.listdir('RSNACOVID/train/'+i+'/'+j):
             im= dicom.dcmread('RSNACOVID/train/'+i+'/'+j+'/'+k)
             print('RSNACOVID/train/'+i+'/'+j+'/'+k)
             im.PhotometricInterpretation = 'YBR_FULL'
             im=im.pixel_array.astype(float)
             rescaled_image=(np.maximum(im,0)/im.max())*255
             final_image=np.uint8(rescaled_image)
             final_image= Image.fromarray(final_image)
             if int(df.loc[df['ID']==i+'_study']['Labels'])==2:
                 final_image.save('/home/shahad.hardan/Downloads/ML Pro Dataset/Negative/'+i+'.png')
             elif int(df.loc[df['ID']==i+'_study']['Labels'])==0:
                 final_image.save('/home/shahad.hardan/Downloads/ML Pro Dataset/Atypical Appearance/'+i+'.png')
             elif int(df.loc[df['ID']==i+'_study']['Labels'])==1:
                 final_image.save('/home/shahad.hardan/Downloads/ML Pro Dataset/Indeterminate/'+i+'.png')
             elif int(df.loc[df['ID']==i+'_study']['Labels'])==3:
                 final_image.save('/home/shahad.hardan/Downloads/ML Pro Dataset/Typical Appearance/'+i+'.png')

#Resize image Negative
for i in os.listdir('/home/shahad.hardan/Downloads/ML Pro Dataset/Negative/'):
    im = Image.open('/home/shahad.hardan/Downloads/ML Pro Dataset/Negative/'+i)
    new_im= im.resize((224,224))
    new_im.save('/home/shahad.hardan/Downloads/COVID_resized 224/Negative /'+i)

# Resize image Typical Appearance
for i in os.listdir('/home/shahad.hardan/Downloads/ML Pro Dataset/Typical Appearance/'):
    im = Image.open('/home/shahad.hardan/Downloads/ML Pro Dataset/Typical Appearance/'+i)
    new_im= im.resize((224,224))
    new_im.save('/home/shahad.hardan/Downloads/COVID_resized 224/Typical Appearance/'+i)

#Transform gray scale atypical images to RGB
 path_org= '/home/shahad.hardan/Downloads/COVID_resized 224/Atypical Appearance/'
 for i in os.listdir(path_org):
     image=io.imread(path_org+i)
     path='/home/shahad.hardan/Downloads/COVID_RGB/Atypical/'
     if len(image.shape)==3:
         path = path + (i)
         io.imsave(path, image)
     elif len(image.shape)==2:
         image=color.gray2rgb(image)
         path = path + (i)
         io.imsave(path, image)

#Transform gray scale typical images to RGB
path_org= '/home/shahad.hardan/Downloads/COVID_resized 224/Typical Appearance/'
for i in os.listdir(path_org):
     image=io.imread(path_org+i)
     path='/home/shahad.hardan/Downloads/COVID_RGB/Typical/'
     if len(image.shape)==3:
         path = path + (i)
         io.imsave(path, image)
     elif len(image.shape)==2:
         image=color.gray2rgb(image)
         path = path + (i)
         io.imsave(path, image)

#Transform gray scale negative images to RGB
path_org= '/home/shahad.hardan/Downloads/COVID_resized 224/Negative /'
for i in os.listdir(path_org):
     image=io.imread(path_org+i)
     path='/home/shahad.hardan/Downloads/COVID_RGB/Negative/'
     if len(image.shape)==3:
         path = path + (i)
         io.imsave(path, image)
     elif len(image.shape)==2:
         image=color.gray2rgb(image)
         path = path + (i)
         io.imsave(path, image)

#Transform gray scale indeterminate images to RGB
path_org= '/home/shahad.hardan/Downloads/COVID_resized 224/Indeterminate/'
for i in os.listdir(path_org):
     image=io.imread(path_org+i)
     path='/home/shahad.hardan/Downloads/COVID_RGB/Indeterminate/'
     if len(image.shape)==3:
         path = path + (i)
         io.imsave(path, image)
     elif len(image.shape)==2:
         image=color.gray2rgb(image)
         path = path + (i)
         io.imsave(path, image)


