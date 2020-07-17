import cv2,os,re
import numpy as np
import matplotlib.pyplot as plt
import maxflow 
from PIL import Image
from PIL import ImageEnhance
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import imutils
import shutil, en_core_web_sm
import spacy
from spacy.matcher import Matcher
import sys

Input_Path=r'E:\After_corona\Bill Project\Restaurant_bills'
Working_Path=r'E:\After_corona\Bill Project\Working_Path'
Procees_Path=r'E:\After_corona\Bill Project\Processing_Path'
OutPut_Path=r'E:\After_corona\Bill Project\Output_Path'
Final_Input=r'E:\After_corona\Bill Project\Processed_Image'
SpacyPath = r'E:\en_core_web_sm\en_core_web_sm-2.2.0'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'   
nlp = en_core_web_sm.load()
ORG='ORG'
Person='PERSON'
Location='GPE'
nlp = spacy.load(SpacyPath)

def Organization(PageText):
    PageText = str(PageText)
    doc = nlp(PageText) 
      
    for ent in doc.ents: 
        if ent.label_==ORG:
            text=str(ent.text)                
            return text  
            
def Location(PageText):
    PageText = str(PageText)
    doc = nlp(PageText) 
    for ent in doc.ents: 
        if ent.label_==Location:
            text=str(ent.text)                
            return text    
    
def Date_Detection(text):
    text = str(text)
    doc = nlp(text)
   
    for match in re.finditer(r'\d+\W+\d+\W+\d+', doc.text):
        start, end = match.span()
        span = doc.char_span(start, end)
       
        if span is not None:
           return text    
def cleanOutput(out):
                    out=  os.linesep.join([s for s in  out.splitlines() if s])
                    out = (re.sub(r'\n\s*\n','\n',out,re.MULTILINE))
                    out = (re.sub(r';','', out ))
                    out = (re.sub(r';','',  out ))
                    out=re.sub(r"\n"," ",out)
                    out= re.sub(r'\s{2}', '', out)
                    return out           
def removefromend(string):  
            endString=string           
            endString=re.sub(r'([^\w\s]|_)+(?=\s|$)', '', endString)
            return endString
# ********************************************************************************************************************************
                 # IMAGE PREPROCESSING USING OPENCV,CV2,PIL,MAXFLOW
# ********************************************************************************************************************************
           
image_Path=os.listdir(Input_Path)
# Important parameter
# Higher values means making the image smoother
smoothing = -5.0
for index in image_Path:
    S_No=1
    img = cv2.imread(Input_Path+'\\'+str(index)) #Input Image Path  
    
    #Scaling (Image Large)
    large = cv2.resize(img, (0,0), fx=1.5, fy=1.5)
    cv2.imwrite(Working_Path+'\\'+'Scaled_'+index,large)      
    img = cv2.imread(Working_Path+'\\'+'Scaled_'+index)
   
    # Load the image and convert it to grayscale image 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = 255 * (img > 112).astype(np.uint8)
    
    # Create the graph.
    g = maxflow.Graph[int]()
    
    # Add the nodes. nodeids has the identifiers of the nodes in the grid.
    nodeids = g.add_grid_nodes(img.shape)
   
    # Add non-terminal edges with the same capacity.
    g.add_grid_edges(nodeids, smoothing)
    
    # Add the terminal edges. The image pixels are the capacities
    # of the edges from the source node. The inverted image pixels
    # are the capacities of the edges to the sink node.
    g.add_grid_tedges(nodeids, img, 255-img)
    
    # Find the maximum flow.
    g.maxflow()
    # Get the segments of the nodes in the grid.
    sgm = g.get_grid_segments(nodeids)
    
    # The labels should be 1 where sgm is False and 0 otherwise.
    img_denoised = np.logical_not(sgm).astype(np.uint8) * 255
    
    # Save denoised image
    cv2.imwrite(Working_Path+'\\'+'Denoised_'+index, img_denoised)
    
    # Load image, grayscale, Gaussian blur, adaptive threshold
    img = cv2.imread(Working_Path+'\\'+'Denoised_'+index)
    ret,thresh = cv2.threshold(img,55,255,cv2.THRESH_BINARY)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)))
    cv2.imwrite(Procees_Path+'\\'+'Enhancexd_'+index, opening)
    
    #Enhance The Image
    image = Image.open(Procees_Path+'\\'+'Enhancexd_'+index)
    enh_sha = ImageEnhance.Sharpness(image)
    sharpness = 100
    image_sharped = enh_sha.enhance(sharpness)
    image4 = np.array(image_sharped)
    cv2.imwrite(Final_Input+'\\'+'Final_'+index, image4)

# ********************************************************************************************************************************    
    #OCR Part
# ********************************************************************************************************************************
    
    im = Image.open(Final_Input+'\\'+'Final_'+index) # the second one 
    im = im.filter(ImageFilter.ModeFilter()) #Filter The Image
    enhancer = ImageEnhance.Sharpness(im)
    im = enhancer.enhance(4)
    im = im.convert('1')
    im.save('temp2.jpg')
    text = pytesseract.image_to_string(Image.open('temp2.jpg'))
    print('*****************************************************************\n')
    print(index,'_Processed','\n',text,'****************************************************************','\n')
    
# ********************************************************************************************************************************
         #INFROMATION EXTRACTION USING NATURAL LANGUAGE PROCESSING AND REGEX
# ******************************************************************************************************************************** 
       
    # Find the Date Using Regular Expression 
    Date=re.findall(r'\d{2}\W+\d{2}\W+\d{2}',text)
    Date=''.join(Date)
    Date =cleanOutput(str(Date))
    Date =removefromend(str(Date))
    print(index,'_Date:',Date)
   
    Hotel_name=Organization(text)
    Hotel_name =cleanOutput(str(Hotel_name))
    Hotel_name =removefromend(str(Hotel_name))
    Hotel_name=''.join(Hotel_name)
    print('\n Hotel Name:',Hotel_name)
    
    Location_Name=Location(text)
    Location_Name =cleanOutput(str(Location_Name))
    Location_Name=''.join(Location_Name)
    Location_Name =removefromend(str(Location_Name))
    print('\n Hotel Location:',Location_Name)
    
    Bill_NO=re.findall(r'Bill No\W+\d+\W+\d+|No\W+\d+',text)
    Bill_NO=''.join(Bill_NO)
    Bill_NO=re.sub(r'Bill No','',Bill_NO)
    Bill_NO =cleanOutput(str(Bill_NO))
    Bill_NO =removefromend(str(Bill_NO))
    print(index,'Bill_Number:',Bill_NO)
   
    Grand_Total=re.findall(r'Grand Total\W+\d+|CASH\s+\d+\.\d+',text)
    Grand_Total=''.join(Grand_Total)
    Grand_Total=re.sub(r'Grand Total','',Grand_Total)
    Grand_Total =cleanOutput(str(Grand_Total))
    Grand_Total =removefromend(str(Grand_Total))
    print(index,'Grand_Total:',Grand_Total)
    
# ********************************************************************************************************************************
         # OUTPUT WRITE IN FILE
# ********************************************************************************************************************************     
    index1=re.sub('.png','',str(index))

    FileOut1 = open(OutPut_Path+'\\'+str(index1)+'.csv',"w",encoding='UTF-7' )
    FileOut1.write('Serial Nos'+"|"+'File Name'+"|"+'Data Field Name'+"|"+"Field Value"+"|"+"\n")
    FileOut1.write(str(S_No)+"|"+str(index1)+"|"+"Hotel Name"+"|"+str(Hotel_name)+"\n")
    S_No+=1
    FileOut1.write(str(S_No)+"|"+str(index1)+"|"+"Date"+"|"+str(Date)+"\n")
    S_No+=1
    FileOut1.write(str(S_No)+"|"+str(index1)+"|"+"Location"+"|"+str(Location_Name)+"\n")
    S_No+=1
    FileOut1.write(str(S_No)+"|"+str(index1)+"|"+"Bill N0"+"|"+str(Bill_NO)+"\n")
    S_No+=1
    FileOut1.write(str(S_No)+"|"+str(index1)+"|"+"Grand_Total/Cash"+"|"+str(Grand_Total)+"\n")
    S_No+=1
    FileOut1.close()