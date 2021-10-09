###This file contains the utility functions that are commonly used in EDER_VIANN packages

import csv
import math
import os, shutil
import subprocess
#import time as t
import datetime
import ast
import numpy as np
#import scipy
import copy
#from scipy.stats import chisquare

#This utility provides Timestamps for print messages
def TimeStamp():
 return "["+datetime.datetime.now().strftime("%D")+' '+datetime.datetime.now().strftime("%H:%M:%S")+"]"

class Seed:
      def __init__(self,tracks):
          self.TrackHeader=sorted(tracks, key=str.lower)
          self.Multiplicity=len(self.TrackHeader)
      def __eq__(self, other):
        return ('-'.join(self.TrackHeader)) == ('-'.join(other.TrackHeader))
      def __hash__(self):
        return hash(('-'.join(self.TrackHeader)))
      def DecorateTracks(self,RawHits): #Decorate hit information
          self.TrackHits=[]
          for s in range(len(self.TrackHeader)):
              self.TrackHits.append([])
              for t in RawHits:
                   if self.TrackHeader[s]==t[3]:
                      self.TrackHits[s].append(t[:3])
          for Hit in range(0, len(self.TrackHits)):
             self.TrackHits[Hit]=sorted(self.TrackHits[Hit],key=lambda x: float(x[2]),reverse=False)

      def DecorateSeedGeoInfo(self):
          if hasattr(self,'TrackHits'):
             if self.Multiplicity==2:
                __XZ1=Seed.GetEquationOfTrack(self.TrackHits[0])[0]
                __XZ2=Seed.GetEquationOfTrack(self.TrackHits[1])[0]
                __YZ1=Seed.GetEquationOfTrack(self.TrackHits[0])[1]
                __YZ2=Seed.GetEquationOfTrack(self.TrackHits[1])[1]
                __X1S=Seed.GetEquationOfTrack(self.TrackHits[0])[3]
                __X2S=Seed.GetEquationOfTrack(self.TrackHits[1])[3]
                __Y1S=Seed.GetEquationOfTrack(self.TrackHits[0])[4]
                __Y2S=Seed.GetEquationOfTrack(self.TrackHits[1])[4]
                __Z1S=Seed.GetEquationOfTrack(self.TrackHits[0])[5]
                __Z2S=Seed.GetEquationOfTrack(self.TrackHits[1])[5]
                __vector_1_st = np.array([np.polyval(__XZ1,self.TrackHits[0][0][2]),np.polyval(__YZ1,self.TrackHits[0][0][2]),self.TrackHits[0][0][2]])
                __vector_1_end = np.array([np.polyval(__XZ1,self.TrackHits[0][len(self.TrackHits[0])-1][2]),np.polyval(__YZ1,self.TrackHits[0][len(self.TrackHits[0])-1][2]),self.TrackHits[0][len(self.TrackHits[0])-1][2]])
                __vector_2_st = np.array([np.polyval(__XZ2,self.TrackHits[0][0][2]),np.polyval(__YZ2,self.TrackHits[0][0][2]),self.TrackHits[0][0][2]])
                __vector_2_end = np.array([np.polyval(__XZ2,self.TrackHits[0][len(self.TrackHits[0])-1][2]),np.polyval(__YZ2,self.TrackHits[0][len(self.TrackHits[0])-1][2]),self.TrackHits[0][len(self.TrackHits[0])-1][2]])
                __result=Seed.closestDistanceBetweenLines(__vector_1_st,__vector_1_end,__vector_2_st,__vector_2_end,clampAll=False,clampA0=False,clampA1=False,clampB0=False,clampB1=False)
                __midpoint=(__result[0]+__result[1])/2
                __D1M=math.sqrt(((__midpoint[0]-__X1S)**2) + ((__midpoint[1]-__Y1S)**2) + ((__midpoint[2]-__Z1S)**2))
                __D2M=math.sqrt(((__midpoint[0]-__X2S)**2) + ((__midpoint[1]-__Y2S)**2) + ((__midpoint[2]-__Z2S)**2))
                __v1=np.subtract(__vector_1_end,__midpoint)
                __v2=np.subtract(__vector_2_end,__midpoint)
                self.angle=Seed.angle_between(__v1, __v2)
                self.Vx=__midpoint[0]
                self.Vy=__midpoint[1]
                self.Vz=__midpoint[2]
                self.DOCA=__result[2]
                self.V_Tr=[__D1M,__D2M]
                self.Tr_Tr=math.sqrt(((float(self.TrackHits[0][0][0])-float(self.TrackHits[1][0][0]))**2)+((float(self.TrackHits[0][0][1])-float(self.TrackHits[1][0][1]))**2)+((float(self.TrackHits[0][0][2])-float(self.TrackHits[1][0][2]))**2))
             else:
                 raise ValueError("Method 'DecorateSeedGeoInfo' currently works for seeds with track multiplicity of 2 only")
          else:
                raise ValueError("Method 'DecorateSeedGeoInfo' works only if 'DecorateTracks' method has been acted upon the seed before")

      def SeedQualityCheck(self,VO_min_Z,VO_max_Z,MaxDoca,TV_Min_Dist):
                    self.GeoFit = (self.DOCA<=MaxDoca and min(self.V_Tr)<=TV_Min_Dist and self.Vz>=VO_min_Z and self.Vz<=VO_max_Z)

      def CNNFitSeed(self,Prediction):
          self.Seed_CNN_Fit=Prediction

      def AssignCNNVxId(self,ID):
          self.VX_CNN_ID=ID

      def InjectSeed(self,OtherSeed):
          __overlap=False
          for t1 in self.TrackHeader:
              for t2 in OtherSeed.TrackHeader:
                  if t1==t2:
                      __overlap=True
                      break
          if __overlap:
              overlap_matrix=[]
              for t1 in range(len(self.TrackHeader)):
                 for t2 in range(len(OtherSeed.TrackHeader)):
                    if self.TrackHeader[t1]==OtherSeed.TrackHeader[t2]:
                       overlap_matrix.append(t2)

              for t2 in range(len(OtherSeed.TrackHeader)):
                if (t2 in overlap_matrix)==False:
                  self.TrackHeader.append(OtherSeed.TrackHeader[t2])
                  if hasattr(self,'TrackHits') and hasattr(OtherSeed,'TrackHits'):
                          self.TrackHits.append(OtherSeed.TrackHits[t2])
                  if hasattr(self,'MC_truth_label') and hasattr(OtherSeed,'MC_truth_label'):
                         self.MC_truth_label=(self.MC_truth_label and OtherSeed.MC_truth_label)
                  if hasattr(self,'VX_CNN_Fit') and hasattr(OtherSeed,'VX_CNN_Fit'):
                        self.VX_CNN_Fit+=OtherSeed.VX_CNN_Fit
                  elif hasattr(self,'VX_CNN_Fit'):
                       self.VX_CNN_Fit.append(OtherSeed.Seed_CNN_Fit)
                  elif hasattr(OtherSeed,'VX_CNN_Fit'):
                       self.VX_CNN_Fit=[self.Seed_CNN_Fit]
                       self.VX_CNN_Fit+=OtherSeed.VX_CNN_Fit
                  else:
                      self.VX_CNN_Fit=[]
                      self.VX_CNN_Fit.append(self.Seed_CNN_Fit)
                      self.VX_CNN_Fit.append(OtherSeed.Seed_CNN_Fit)
                  if hasattr(self,'VX_x') and hasattr(OtherSeed,'VX_x'):
                      self.VX_x+=OtherSeed.VX_x
                  elif hasattr(self,'VX_x'):
                       self.VX_x.append(OtherSeed.Vx)
                  elif hasattr(OtherSeed,'VX_x'):
                       self.VX_x=[self.Vx]
                       self.VX_x+=OtherSeed.VX_x
                  else:
                      self.VX_x=[]
                      self.VX_x.append(self.Vx)
                      self.VX_x.append(OtherSeed.Vx)
                          
                  if hasattr(self,'VX_y') and hasattr(OtherSeed,'VX_y'):
                      self.VX_y+=OtherSeed.VX_y
                  elif hasattr(self,'VX_y'):
                       self.VX_y.append(OtherSeed.Vy)
                  elif hasattr(OtherSeed,'VX_y'):
                       self.VX_y=[self.Vy]
                       self.VX_y+=OtherSeed.VX_y
                  else:
                      self.VX_y=[]
                      self.VX_y.append(self.Vy)
                      self.VX_y.append(OtherSeed.Vy)
                  
                  if hasattr(self,'VX_z') and hasattr(OtherSeed,'VX_z'):
                      self.VX_z+=OtherSeed.VX_z
                  elif hasattr(self,'VX_z'):
                       self.VX_z.append(OtherSeed.Vz)
                  elif hasattr(OtherSeed,'VX_z'):
                       self.VX_z=[self.Vz]
                       self.VX_z+=OtherSeed.VX_z
                  else:
                      self.VX_z=[]
                      self.VX_z.append(self.Vz)
                      self.VX_z.append(OtherSeed.Vz)
              self.Multiplicity=len(self.TrackHeader)
              self.Seed_CNN_Fit=sum(self.VX_CNN_Fit)/len(self.VX_CNN_Fit)
              self.Vx=sum(self.VX_x)/len(self.VX_x)
              self.Vy=sum(self.VX_y)/len(self.VX_y)
              self.Vz=sum(self.VX_z)/len(self.VX_z)
              if hasattr(self,'angle'):
                  delattr(self,'angle')
              if hasattr(self,'DOCA'):
                  delattr(self,'DOCA')
              if hasattr(self,'V_Tr'):
                  delattr(self,'V_Tr')
              if hasattr(self,'Tr_Tr'):
                  delattr(self,'Tr_Tr')



              return True

          else:
              return __overlap


      def MCtruthClassifySeed(self,label):
          self.MC_truth_label=label

      def PrepareTrackPrint(self,MaxX,MaxY,MaxZ,Res,Rescale):
          __TempTrack=copy.deepcopy(self.TrackHits)
          self.Resolution=Res
          self.bX=int(round(MaxX/self.Resolution,0))
          self.bY=int(round(MaxY/self.Resolution,0))
          self.bZ=int(round(MaxZ/self.Resolution,0))
          self.H=(self.bX)*2
          self.W=(self.bY)*2
          self.L=(self.bZ)
          __LongestDistance=0.0
          for __Track in __TempTrack:
            __Xdiff=float(__Track[len(__Track)-1][0])-float(__Track[0][0])
            __Ydiff=float(__Track[len(__Track)-1][1])-float(__Track[0][1])
            __Zdiff=float(__Track[len(__Track)-1][2])-float(__Track[0][2])
            __Distance=math.sqrt(__Xdiff**2+__Ydiff**2+__Zdiff**2)
            if __Distance>=__LongestDistance:
                __LongestDistance=__Distance
                __FinX=float(__Track[0][0])
                __FinY=float(__Track[0][1])
                __FinZ=float(__Track[0][2])
                self.LongestTrackInd=__TempTrack.index(__Track)
          for __Tracks in __TempTrack:
              for __Hits in __Tracks:
                  __Hits[0]=float(__Hits[0])-__FinX
                  __Hits[1]=float(__Hits[1])-__FinY
                  __Hits[2]=float(__Hits[2])-__FinZ

          #Lon Rotate x
          __LongestDistance=0.0
          __Track=__TempTrack[self.LongestTrackInd]
          __Vardiff=float(__Track[len(__Track)-1][0])
          __Zdiff=float(__Track[len(__Track)-1][2])
          __vector_1 = [__Zdiff, 0]
          __vector_2 = [__Zdiff, __Vardiff]
          __Angle=Seed.angle_between(__vector_1, __vector_2)
          if np.isnan(__Angle)==True:
                    __Angle=0.0
          for __Tracks in __TempTrack:
            for __hits in __Tracks:
                __Z=float(__hits[2])
                __Pos=float(__hits[0])
                __hits[2]=(__Z*math.cos(-__Angle)) - (__Pos * math.sin(-__Angle))
                __hits[0]=(__Z*math.sin(-__Angle)) + (__Pos * math.cos(-__Angle))
          #Lon Rotate y
          __LongestDistance=0.0
          __Track=__TempTrack[self.LongestTrackInd]
          __Vardiff=float(__Track[len(__Track)-1][1])
          __Zdiff=float(__Track[len(__Track)-1][2])
          __vector_1 = [__Zdiff, 0]
          __vector_2 = [__Zdiff, __Vardiff]
          __Angle=Seed.angle_between(__vector_1, __vector_2)
          if np.isnan(__Angle)==True:
                    __Angle=0.0
          for __Tracks in __TempTrack:
           for __hits in __Tracks:
                __Z=float(__hits[2])
                __Pos=float(__hits[1])
                __hits[2]=(__Z*math.cos(-__Angle)) - (__Pos * math.sin(-__Angle))
                __hits[1]=(__Z*math.sin(-__Angle)) + (__Pos * math.cos(-__Angle))
          #Phi rotate print

          __LongestDistance=0.0
          for __Track in __TempTrack:
                __X=float(__Track[len(__Track)-1][0])
                __Y=float(__Track[len(__Track)-1][1])
                __Distance=math.sqrt((__X**2)+(__Y**2))
                if __Distance>=__LongestDistance:
                 __LongestDistance=__Distance
                 __vector_1 = [__Distance, 0]
                 __vector_2 = [__X, __Y]
                 __Angle=-Seed.angle_between(__vector_1,__vector_2)
          if np.isnan(__Angle)==True:
                    __Angle=0.0
          for __Tracks in __TempTrack:
            for __hits in __Tracks:
                __X=float(__hits[0])
                __Y=float(__hits[1])
                __hits[0]=(__X*math.cos(__Angle)) - (__Y * math.sin(__Angle))
                __hits[1]=(__X*math.sin(__Angle)) + (__Y * math.cos(__Angle))

          #After shift

          __FinZ=666666.0
          for __Track in __TempTrack:
              if __FinZ>float(__Track[0][2]):
                 __FinZ=float(__Track[0][2])-self.Resolution
          __FinX=float(__TempTrack[self.LongestTrackInd][0][0])
          __FinY=float(__TempTrack[self.LongestTrackInd][0][1])
          for __Tracks in __TempTrack:
              for __Hits in __Tracks:
                 __Hits[0]=float(__Hits[0])-__FinX
                 __Hits[1]=float(__Hits[1])-__FinY
                 __Hits[2]=float(__Hits[2])-__FinZ

          #Rescale
          if Rescale:
              __X=[]
              __Y=[]
              __Z=[]
              for __Tracks in __TempTrack:
                for __hits in __Tracks:
                    __X.append(__hits[0])
                    __Y.append(__hits[1])
                    __Z.append(__hits[2])
              __dUpX=MaxX-max(__X)
              __dDownX=MaxX+min(__X)
              __dX=(__dUpX+__dDownX)/2
              __xshift=__dUpX-__dX
              __X=[]
              for __Tracks in __TempTrack:
                for __hits in __Tracks:
                    __hits[0]=__hits[0]+__xshift
                    __X.append(__hits[0])
            ##########Y
              __dUpY=MaxY-max(__Y)
              __dDownY=MaxY+min(__Y)
              __dY=(__dUpY+__dDownY)/2
              __yshift=__dUpY-__dY
              __Y=[]
              for __Tracks in __TempTrack:
                for __hits in __Tracks:
                    __hits[1]=__hits[1]+__yshift
                    __Y.append(__hits[1])
              __min_scale=max(max(__X)/(MaxX-(2*self.Resolution)),max(__Y)/(MaxY-(2*self.Resolution)), max(__Z)/(MaxZ-(2*self.Resolution)))
              for __Tracks in __TempTrack:
                for __hits in __Tracks:
                    __hits[0]=int(round(__hits[0]/__min_scale,0))
                    __hits[1]=int(round(__hits[1]/__min_scale,0))
                    __hits[2]=int(round(__hits[2]/__min_scale,0))

          #Enchance track
          __TempEnchTrack=[]
          for __Tracks in __TempTrack:
              for h in range(0,len(__Tracks)-1):
                  __deltaX=float(__Tracks[h+1][0])-float(__Tracks[h][0])
                  __deltaZ=float(__Tracks[h+1][2])-float(__Tracks[h][2])
                  __deltaY=float(__Tracks[h+1][1])-float(__Tracks[h][1])
                  try:
                   __vector_1 = [__deltaZ,0]
                   __vector_2 = [__deltaZ, __deltaX]
                   __ThetaAngle=Seed.angle_between(__vector_1, __vector_2)
                  except:
                    __ThetaAngle=0.0
                  try:
                    __vector_1 = [__deltaZ,0]
                    __vector_2 = [__deltaZ, __deltaY]
                    __PhiAngle=Seed.angle_between(__vector_1, __vector_2)
                  except:
                    __PhiAngle=0.0
                  __TotalDistance=math.sqrt((__deltaX**2)+(__deltaY**2)+(__deltaZ**2))
                  __Distance=(float(self.Resolution)/3)
                  if __Distance>=0 and __Distance<1:
                     __Distance=1.0
                  if __Distance<0 and __Distance>-1:
                     __Distance=-1.0
                  __Iterations=int(round(__TotalDistance/__Distance,0))
                  for i in range(1,__Iterations):
                      __New_Hit=[]
                      if math.isnan(float(__Tracks[h][0])+float(i)*__Distance*math.sin(__ThetaAngle)):
                         continue
                      if math.isnan(float(__Tracks[h][1])+float(i)*__Distance*math.sin(__PhiAngle)):
                         continue
                      if math.isnan(float(__Tracks[h][2])+float(i)*__Distance*math.cos(__ThetaAngle)):
                         continue
                      __New_Hit.append(float(__Tracks[h][0])+float(i)*__Distance*math.sin(__ThetaAngle))
                      __New_Hit.append(float(__Tracks[h][1])+float(i)*__Distance*math.sin(__PhiAngle))
                      __New_Hit.append(float(__Tracks[h][2])+float(i)*__Distance*math.cos(__ThetaAngle))
                      __TempEnchTrack.append(__New_Hit)

          #Pixelise print

          self.TrackPrint=[]
          for __Tracks in __TempTrack:
              for __Hits in __Tracks:
                  __Hits[0]=int(round(float(__Hits[0])/self.Resolution,0))
                  __Hits[1]=int(round(float(__Hits[1])/self.Resolution,0))
                  __Hits[2]=int(round(float(__Hits[2])/self.Resolution,0))
                  self.TrackPrint.append(str(__Hits))
          for __Hits in __TempEnchTrack:
                  __Hits[0]=int(round(float(__Hits[0])/self.Resolution,0))
                  __Hits[1]=int(round(float(__Hits[1])/self.Resolution,0))
                  __Hits[2]=int(round(float(__Hits[2])/self.Resolution,0))
                  self.TrackPrint.append(str(__Hits))
          self.TrackPrint=list(set(self.TrackPrint))
          for p in range(len(self.TrackPrint)):
              self.TrackPrint[p]=ast.literal_eval(self.TrackPrint[p])

          del __TempEnchTrack
          del __TempTrack

      def UnloadTrackPrint(self):
          delattr(self,'TrackPrint')
          delattr(self,'bX')
          delattr(self,'bY')
          delattr(self,'bZ')
          delattr(self,'H')
          delattr(self,'W')
          delattr(self,'L')
          delattr(self,'LongestTrackInd')


      def Plot(self,PlotType):
        if PlotType=='XZ':
          __InitialData=[]
          __Index=-1
          for x in range(-self.bX,self.bX):
             for z in range(0,self.bZ):
                 __InitialData.append(0.0)
          __Matrix = np.array(__InitialData)
          __Matrix=np.reshape(__Matrix,(self.H,self.L))
          for __Hits in self.TrackPrint:
                if abs(int(__Hits[0]))<self.bX and abs(int(__Hits[2]))<self.bZ:
                   __Matrix[int(__Hits[0])+self.bX][int(__Hits[2])]=1
          import matplotlib as plt
          from matplotlib.colors import LogNorm
          from matplotlib import pyplot as plt
          plt.title('Seed '+':'.join(self.TrackHeader))
          plt.xlabel('Z [microns /'+str(int(self.Resolution))+']')
          plt.ylabel('X [microns /'+str(int(self.Resolution))+']')
          __image=plt.imshow(__Matrix,cmap='gray_r',extent=[0,self.bZ,self.bX,-self.bX])
          plt.gca().invert_yaxis()
          plt.show()

        if PlotType=='YZ':
           __InitialData=[]
           __Index=-1
           for y in range(-__bY,__bY):
               for z in range(0,__bZ):
                 __InitialData.append(0.0)

           __Matrix = np.array(__InitialData)
           __Matrix=np.reshape(__Matrix,(__W,__L))
        if PlotType=='XY':
           __InitialData=[]
           __Index=-1
           for x in range(-__bX,__bX):
                 for y in range(-__bY,__bY):
                     __InitialData.append(0.0)
           __Matrix = np.array(__InitialData)
           __Matrix=np.reshape(__Matrix,(__H,__W))

        #if PlotType=='XZ':

      @staticmethod
      def unit_vector(vector):
          return vector / np.linalg.norm(vector)

      def angle_between(v1, v2):
            v1_u = Seed.unit_vector(v1)
            v2_u = Seed.unit_vector(v2)
            dot = v1_u[0]*v2_u[0] + v1_u[1]*v2_u[1]      # dot product
            det = v1_u[0]*v2_u[1] - v1_u[1]*v2_u[0]      # determinant
            return np.arctan2(det, dot)

      def GetEquationOfTrack(Track):
          Xval=[]
          Yval=[]
          Zval=[]
          for Hits in Track:
              Xval.append(Hits[0])
              Yval.append(Hits[1])
              Zval.append(Hits[2])
          XZ=np.polyfit(Zval,Xval,1)
          YZ=np.polyfit(Zval,Yval,1)
          return (XZ,YZ, 'N/A',Xval[0],Yval[0],Zval[0])

      def closestDistanceBetweenLines(a0,a1,b0,b1,clampAll=False,clampA0=False,clampA1=False,clampB0=False,clampB1=False):
            a0=np.array(a0)
            a1=np.array(a1)
            b0=np.array(b0)
            b1=np.array(b1)
            # If clampAll=True, set all clamps to True
            if clampAll:
                clampA0=True
                clampA1=True
                clampB0=True
                clampB1=True


            # Calculate denomitator
            A = a1 - a0
            B = b1 - b0
            magA = np.linalg.norm(A)
            magB = np.linalg.norm(B)

            _A = A / magA
            _B = B / magB

            cross = np.cross(_A, _B);
            denom = np.linalg.norm(cross)**2


            # If lines are parallel (denom=0) test if lines overlap.
            # If they don't overlap then there is a closest point solution.
            # If they do overlap, there are infinite closest positions, but there is a closest distance
            if not denom:
                d0 = np.dot(_A,(b0-a0))

                # Overlap only possible with clamping
                if clampA0 or clampA1 or clampB0 or clampB1:
                    d1 = np.dot(_A,(b1-a0))

                    # Is segment B before A?
                    if d0 <= 0 >= d1:
                        if clampA0 and clampB1:
                            if np.absolute(d0) < np.absolute(d1):
                                return a0,b0,np.linalg.norm(a0-b0)
                            return a0,b1,np.linalg.norm(a0-b1)


                    # Is segment B after A?
                    elif d0 >= magA <= d1:
                        if clampA1 and clampB0:
                            if np.absolute(d0) < np.absolute(d1):
                                return a1,b0,np.linalg.norm(a1-b0)
                            return a1,b1,np.linalg.norm(a1-b1)


                # Segments overlap, return distance between parallel segments
                return None,None,np.linalg.norm(((d0*_A)+a0)-b0)



            # Lines criss-cross: Calculate the projected closest points
            t = (b0 - a0);
            detA = np.linalg.det([t, _B, cross])
            detB = np.linalg.det([t, _A, cross])

            t0 = detA/denom;
            t1 = detB/denom;

            pA = a0 + (_A * t0) # Projected closest point on segment A
            pB = b0 + (_B * t1) # Projected closest point on segment B


            # Clamp projections
            if clampA0 or clampA1 or clampB0 or clampB1:
                if clampA0 and t0 < 0:
                    pA = a0
                elif clampA1 and t0 > magA:
                    pA = a1

                if clampB0 and t1 < 0:
                    pB = b0
                elif clampB1 and t1 > magB:
                    pB = b1

                # Clamp projection A
                if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
                    dot = np.dot(_B,(pA-b0))
                    if clampB0 and dot < 0:
                        dot = 0
                    elif clampB1 and dot > magB:
                        dot = magB
                    pB = b0 + (_B * dot)

                # Clamp projection B
                if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
                    dot = np.dot(_A,(pB-a0))
                    if clampA0 and dot < 0:
                        dot = 0
                    elif clampA1 and dot > magA:
                        dot = magA
                    pA = a0 + (_A * dot)


            return pA,pB,np.linalg.norm(pA-pB)




 #if args.PlotType=='YZ':

 # for Tracks in data[1]:
 #        for Hits in Tracks:
  #               if abs(Hits[1])<boundsY and abs(Hits[2])<boundsZ:
  #                 Matrix[Hits[1]+boundsY][Hits[2]]+=1
 # for Tracks in additional_data[1]:
 #     for Hits in Tracks:
 #        if abs(Hits[1])<boundsY and abs(Hits[2])<boundsZ:
 #         Matrix[Hits[1]+boundsY][Hits[2]]+=1
 #if args.PlotType=='XY':
 # for Tracks in data[1]:
 #    for Hits in Tracks:
 #      if abs(Hits[0])<boundsX and abs(Hits[1])<boundsY:
 #        Matrix[Hits[0]+boundsX][Hits[1]+boundsY]+=1
  #for Tracks in additional_data[1]:
 #     for Hits in Tracks:
  #       if abs(Hits[0])<boundsX and abs(Hits[1])<boundsY:
   #       Matrix[Hits[0]+boundsX][Hits[1]+boundsY]+=1
def CleanFolder(folder,key):
    if key=='':
      for the_file in os.listdir(folder):
                file_path=os.path.join(folder, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)
    else:
      for the_file in os.listdir(folder):
                file_path=os.path.join(folder, the_file)
                try:
                    if os.path.isfile(file_path) and (key in the_file):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)
#This function automates csv read/write operations
def LogOperations(flocation,mode, message):
    if mode=='UpdateLog':
        csv_writer_log=open(flocation,"a")
        log_writer = csv.writer(csv_writer_log)
        if len(message)>0:
         for m in message:
          log_writer.writerow(m)
        csv_writer_log.close()
    if mode=='StartLog':
        csv_writer_log=open(flocation,"w")
        log_writer = csv.writer(csv_writer_log)
        if len(message)>0:
         for m in message:
           log_writer.writerow(m)
        csv_writer_log.close()
    if mode=='ReadLog':
        csv_reader_log=open(flocation,"r")
        log_reader = csv.reader(csv_reader_log)
        return list(log_reader)

#Reconstruction

def BSubmitCreateSeedsJobsCondor(job):
            SHName=job[11]+'/HTCondor/SH/SH_R2_'+str(job[0])+'_'+str(job[1])+'.sh'
            SUBName=job[11]+'/HTCondor/SUB/SUB_R2_'+str(job[0])+'_'+str(job[1])+'.sub'
            #MSGName='MSG_R2_'
            OptionLine=(' --Set '+str(job[0]))
            OptionLine+=(' --Subset $1')
            OptionLine+=(" --EOS "+str(job[12]))
            OptionLine+=(" --AFS "+str(job[11]))
            OptionLine+=(" --PlateZ "+str(job[2]))
            OptionLine+=(" --MaxTracks "+str(job[10]))
            OptionLine+=(" --SI_1 "+str(job[3]))
            OptionLine+=(" --SI_2 "+str(job[4]))
            OptionLine+=(" --SI_3 "+str(job[5]))
            OptionLine+=(" --SI_4 "+str(job[6]))
            OptionLine+=(" --SI_5 "+str(job[7]))
            OptionLine+=(" --SI_6 "+str(job[8]))
            OptionLine+=(" --SI_7 "+str(job[9]))
            #MSGName+=str(job[0])
            #MSGName+='_'
            #MSGName+=str(job[1])
            f = open(SUBName, "w")
            f.write("executable = "+SHName)
            f.write("\n")
            #f.write("output ="+job[11]+"/HTCondor/MSG/"+MSGName+".out")
            #f.write("\n")
            #f.write("error ="+job[11]+"/HTCondor/MSG/"+MSGName+".err")
            #f.write("\n")
            #f.write("log ="+job[11]+"/HTCondor/MSG/"+MSGName+".log")
            f.write("\n")
            f.write('requirements = (CERNEnvironment =!= "qa")')
            f.write("\n")
            f.write('arguments = $(Process)')
            f.write("\n")
            f.write('+SoftUsed = "EDER-VIANN-R2"')
            f.write("\n")
            f.write('transfer_output_files = ""')
            f.write("\n")
            f.write('+JobFlavour = "workday"')
            f.write("\n")
            f.write('queue '+str(job[1]))
            f.write("\n")
            f.close()
            TotalLine='python3 '+job[11]+'/Code/Utilities/R2_GenerateSeeds_Sub.py '+OptionLine
            f = open(SHName, "w")
            f.write("#!/bin/bash")
            f.write("\n")
            f.write("set -ux")
            f.write("\n")
            f.write(TotalLine)
            f.write("\n")
            f.close()
            subprocess.call(['condor_submit',SUBName])
            print(TotalLine," has been successfully submitted")
def SubmitCreateSeedsJobsCondor(job):
            SHName=job[11]+'/HTCondor/SH/SH_R2_'+str(job[0])+'_'+str(job[1])+'.sh'
            SUBName=job[11]+'/HTCondor/SUB/SUB_R2_'+str(job[0])+'_'+str(job[1])+'.sub'
            #MSGName='MSG_R2_'
            OptionLine=(' --Set '+str(job[0]))
            OptionLine+=(" --Subset "+str(job[1]))
            OptionLine+=(" --EOS "+str(job[12]))
            OptionLine+=(" --AFS "+str(job[11]))
            OptionLine+=(" --PlateZ "+str(job[2]))
            OptionLine+=(" --MaxTracks "+str(job[10]))
            OptionLine+=(" --SI_1 "+str(job[3]))
            OptionLine+=(" --SI_2 "+str(job[4]))
            OptionLine+=(" --SI_3 "+str(job[5]))
            OptionLine+=(" --SI_4 "+str(job[6]))
            OptionLine+=(" --SI_5 "+str(job[7]))
            OptionLine+=(" --SI_6 "+str(job[8]))
            OptionLine+=(" --SI_7 "+str(job[9]))
            #MSGName+=str(job[0])
            #MSGName+='_'
            #MSGName+=str(job[1])
            f = open(SUBName, "w")
            f.write("executable = "+SHName)
            f.write("\n")
            #f.write("output ="+job[11]+"/HTCondor/MSG/"+MSGName+".out")
            #f.write("\n")
            #f.write("error ="+job[11]+"/HTCondor/MSG/"+MSGName+".err")
            #f.write("\n")
            #f.write("log ="+job[11]+"/HTCondor/MSG/"+MSGName+".log")
            #f.write("\n")
            f.write('requirements = (CERNEnvironment =!= "qa")')
            f.write("\n")
            f.write('+SoftUsed = "EDER-VIANN-R2"')
            f.write("\n")
            f.write('transfer_output_files = ""')
            f.write("\n")
            f.write('+JobFlavour = "workday"')
            f.write("\n")
            f.write('queue 1')
            f.write("\n")
            f.close()
            TotalLine='python3 '+job[11]+'/Code/Utilities/R2_GenerateSeeds_Sub.py '+OptionLine
            f = open(SHName, "w")
            f.write("#!/bin/bash")
            f.write("\n")
            f.write("set -ux")
            f.write("\n")
            f.write(TotalLine)
            f.write("\n")
            f.close()
            subprocess.call(['condor_submit',SUBName])
            print(TotalLine," has been successfully submitted")


def BSubmitFilterSeedsJobsCondor(job):
            SHName=job[7]+'/HTCondor/SH/SH_R3_'+str(job[0])+'_'+str(job[1])+'_'+str(job[2])+'.sh'
            SUBName=job[7]+'/HTCondor/SUB/SUB_R3_'+str(job[0])+'_'+str(job[1])+'_'+str(job[2])+'.sub'
            #MSGName='MSG_R3_'
            OptionLine=(' --Set '+str(job[0]))
            OptionLine+=(" --SubSet "+str(job[1]))
            OptionLine+= " --Fraction $1"
            OptionLine+=(" --EOS "+str(job[8]))
            OptionLine+=(" --AFS "+str(job[7]))
            OptionLine+=(" --VO_T "+str(job[3]))
            OptionLine+=(" --VO_max_Z "+str(job[4]))
            OptionLine+=(" --VO_min_Z "+str(job[5]))
            OptionLine+=(" --MaxDoca "+str(job[6]))
            #MSGName+=str(job[0])
            #MSGName+='_'
            #MSGName+=str(job[1])
            #MSGName+='_'
            #MSGName+=str(job[2])
            f = open(SUBName, "w")
            f.write("executable = "+SHName)
            f.write("\n")
            #f.write("output ="+job[7]+"/HTCondor/MSG/"+MSGName+".out")
            #f.write("\n")
            #f.write("error ="+job[7]+"/HTCondor/MSG/"+MSGName+".err")
            #f.write("\n")
            #f.write("log ="+job[7]+"/HTCondor/MSG/"+MSGName+".log")
            #f.write("\n")
            f.write('requirements = (CERNEnvironment =!= "qa")')
            f.write("\n")
            f.write('arguments = $(Process)')
            f.write("\n")
            f.write('+SoftUsed = "EDER-VIANN-R3"')
            f.write("\n")
            f.write('transfer_output_files = ""')
            f.write("\n")
            f.write('+JobFlavour = "workday"')
            f.write("\n")
            f.write('queue '+str(job[2]))
            f.write("\n")
            f.close()
            TotalLine='python3 '+job[7]+'/Code/Utilities/R3_FilterSeeds_Sub.py '+OptionLine
            f = open(SHName, "w")
            f.write("#!/bin/bash")
            f.write("\n")
            f.write("set -ux")
            f.write("\n")
            f.write(TotalLine)
            f.write("\n")
            f.close()
            subprocess.call(['condor_submit',SUBName])
            print(TotalLine," has been successfully submitted")

def SubmitFilterSeedsJobsCondor(job):
            SHName=job[7]+'/HTCondor/SH/SH_R3_'+str(job[0])+'_'+str(job[1])+'_'+str(job[2])+'.sh'
            SUBName=job[7]+'/HTCondor/SUB/SUB_R3_'+str(job[0])+'_'+str(job[1])+'_'+str(job[2])+'.sub'
            #MSGName='MSG_R3_'
            OptionLine=(' --Set '+str(job[0]))
            OptionLine+=(" --SubSet "+str(job[1]))
            OptionLine+=(" --Fraction "+str(job[2]))
            OptionLine+=(" --EOS "+str(job[8]))
            OptionLine+=(" --AFS "+str(job[7]))
            OptionLine+=(" --VO_T "+str(job[3]))
            OptionLine+=(" --VO_max_Z "+str(job[4]))
            OptionLine+=(" --VO_min_Z "+str(job[5]))
            OptionLine+=(" --MaxDoca "+str(job[6]))
            #MSGName+=str(job[0])
            #MSGName+='_'
            #MSGName+=str(job[1])
            #MSGName+='_'
            #MSGName+=str(job[2])
            f = open(SUBName, "w")
            f.write("executable = "+SHName)
            f.write("\n")
            #f.write("output ="+job[7]+"/HTCondor/MSG/"+MSGName+".out")
            #f.write("\n")
            #f.write("error ="+job[7]+"/HTCondor/MSG/"+MSGName+".err")
            #f.write("\n")
            #f.write("log ="+job[7]+"/HTCondor/MSG/"+MSGName+".log")
            #f.write("\n")
            f.write('requirements = (CERNEnvironment =!= "qa")')
            f.write("\n")
            f.write('+SoftUsed = "EDER-VIANN-R3"')
            f.write("\n")
            f.write('transfer_output_files = ""')
            f.write("\n")
            f.write('+JobFlavour = "workday"')
            f.write("\n")
            f.write('queue 1')
            f.write("\n")
            f.close()
            TotalLine='python3 '+job[7]+'/Code/Utilities/R3_FilterSeeds_Sub.py '+OptionLine
            f = open(SHName, "w")
            f.write("#!/bin/bash")
            f.write("\n")
            f.write("set -ux")
            f.write("\n")
            f.write(TotalLine)
            f.write("\n")
            f.close()
            subprocess.call(['condor_submit',SUBName])
            print(TotalLine," has been successfully submitted")

def BSubmitVertexSeedsJobsCondor(job):
            SHName=job[7]+'/HTCondor/SH/SH_R4_'+str(job[0])+'_'+str(job[1])+'.sh'
            SUBName=job[7]+'/HTCondor/SUB/SUB_R4_'+str(job[0])+'_'+str(job[1])+'.sub'
            #MSGName='MSG_R4_'
            OptionLine=(' --Set '+str(job[0]))
            OptionLine+=(" --Fraction $1")
            OptionLine+=(" --EOS "+str(job[8]))
            OptionLine+=(" --AFS "+str(job[7]))
            OptionLine+=(" --resolution "+str(job[2]))
            OptionLine+=(" --acceptance "+str(job[3]))
            OptionLine+=(" --MaxX "+str(job[4]))
            OptionLine+=(" --MaxY "+str(job[5]))
            OptionLine+=(" --MaxZ "+str(job[6]))
            OptionLine+=(" --ModelName "+str(job[9]))
            #MSGName+=str(job[0])
            #MSGName+='_'
            #MSGName+=str(job[1])
            f = open(SUBName, "w")
            f.write("executable = "+SHName)
            f.write("\n")
            #f.write("output ="+job[7]+"/HTCondor/MSG/"+MSGName+".out")
            #f.write("\n")
            #f.write("error ="+job[7]+"/HTCondor/MSG/"+MSGName+".err")
            #f.write("\n")
            #f.write("log ="+job[7]+"/HTCondor/MSG/"+MSGName+".log")
            #f.write("\n")
            f.write('requirements = (CERNEnvironment =!= "qa")')
            f.write("\n")
            f.write('arguments = $(Process)')
            f.write("\n")
            f.write('+SoftUsed = "EDER-VIANN-R4"')
            f.write("\n")
            f.write('transfer_output_files = ""')
            f.write("\n")
            f.write('+JobFlavour = "workday"')
            f.write("\n")
            f.write('queue '+str(job[1]))
            f.write("\n")
            f.close()
            TotalLine='python3 '+job[7]+'/Code/Utilities/R4_VertexSeeds_Sub.py '+OptionLine
            f = open(SHName, "w")
            f.write("#!/bin/bash")
            f.write("\n")
            f.write("set -ux")
            f.write("\n")
            f.write(TotalLine)
            f.write("\n")
            f.close()
            subprocess.call(['condor_submit',SUBName])
            print(TotalLine," has been successfully submitted")
def SubmitVertexSeedsJobsCondor(job):
            SHName=job[7]+'/HTCondor/SH/SH_R4_'+str(job[0])+'_'+str(job[1])+'.sh'
            SUBName=job[7]+'/HTCondor/SUB/SUB_R4_'+str(job[0])+'_'+str(job[1])+'.sub'
            #MSGName='MSG_R4_'
            OptionLine=(' --Set '+str(job[0]))
            OptionLine+=(" --Fraction "+str(job[1]))
            OptionLine+=(" --EOS "+str(job[8]))
            OptionLine+=(" --AFS "+str(job[7]))
            OptionLine+=(" --resolution "+str(job[2]))
            OptionLine+=(" --acceptance "+str(job[3]))
            OptionLine+=(" --MaxX "+str(job[4]))
            OptionLine+=(" --MaxY "+str(job[5]))
            OptionLine+=(" --MaxZ "+str(job[6]))
            OptionLine+=(" --ModelName "+str(job[9]))
            #MSGName+=str(job[0])
            #MSGName+='_'
            #MSGName+=str(job[1])
            f = open(SUBName, "w")
            f.write("executable = "+SHName)
            f.write("\n")
            #f.write("output ="+job[7]+"/HTCondor/MSG/"+MSGName+".out")
            #f.write("\n")
            #f.write("error ="+job[7]+"/HTCondor/MSG/"+MSGName+".err")
            #f.write("\n")
            #f.write("log ="+job[7]+"/HTCondor/MSG/"+MSGName+".log")
            #f.write("\n")
            f.write('requirements = (CERNEnvironment =!= "qa")')
            f.write("\n")
            f.write('+SoftUsed = "EDER-VIANN-R4"')
            f.write("\n")
            f.write('transfer_output_files = ""')
            f.write("\n")
            f.write('+JobFlavour = "workday"')
            f.write("\n")
            f.write('queue 1')
            f.write("\n")
            f.close()
            TotalLine='python3 '+job[7]+'/Code/Utilities/R4_VertexSeeds_Sub.py '+OptionLine
            f = open(SHName, "w")
            f.write("#!/bin/bash")
            f.write("\n")
            f.write("set -ux")
            f.write("\n")
            f.write(TotalLine)
            f.write("\n")
            f.close()
            subprocess.call(['condor_submit',SUBName])
            print(TotalLine," has been successfully submitted")
#Evaluation
def SubmitCreateEvalSeedsJobsCondor(job):
            SHName=job[2]+'/HTCondor/SH/SH_E2_'+str(job[0])+'.sh'
            SUBName=job[2]+'/HTCondor/SUB/SUB_E2_'+str(job[0])+'.sub'
            #MSGName='MSG_E2_'
            OptionLine=(" --Subset "+str(job[0]))
            OptionLine+=(" --EOS "+str(job[3]))
            OptionLine+=(" --AFS "+str(job[2]))
            OptionLine+=(" --MaxTracks "+str(job[1]))
            #MSGName+=str(job[0])
            f = open(SUBName, "w")
            f.write("executable = "+SHName)
            f.write("\n")
            #f.write("output ="+job[2]+"/HTCondor/MSG/"+MSGName+".out")
            #f.write("\n")
            #f.write("error ="+job[2]+"/HTCondor/MSG/"+MSGName+".err")
            #f.write("\n")
            #f.write("log ="+job[2]+"/HTCondor/MSG/"+MSGName+".log")
            #f.write("\n")
            f.write('requirements = (CERNEnvironment =!= "qa")')
            f.write("\n")
            f.write('+SoftUsed = "EDER-VIANN-E2"')
            f.write("\n")
            f.write('transfer_output_files = ""')
            f.write("\n")
            f.write('+JobFlavour = "workday"')
            f.write("\n")
            f.write('queue 1')
            f.write("\n")
            f.close()
            TotalLine='python3 '+job[2]+'/Code/Utilities/E2_GenerateEvalSeeds_Sub.py '+OptionLine
            f = open(SHName, "w")
            f.write("#!/bin/bash")
            f.write("\n")
            f.write("set -ux")
            f.write("\n")
            f.write(TotalLine)
            f.write("\n")
            f.close()
            subprocess.call(['condor_submit',SUBName])
            print(TotalLine," has been successfully submitted")
def SubmitDecorateSeedsJobsCondor(job):
            SHName=job[2]+'/HTCondor/SH/SH_E3_'+str(job[0])+'_'+str(job[1])+'.sh'
            SUBName=job[2]+'/HTCondor/SUB/SUB_E3_'+str(job[0])+'_'+str(job[1])+'.sub'
            #MSGName='MSG_DC_'
            OptionLine=(' --SubSet '+str(job[0]))
            OptionLine+=(" --Fraction "+str(job[1]))
            OptionLine+=(" --EOS "+str(job[3]))
            OptionLine+=(" --AFS "+str(job[2]))
            #MSGName+=str(job[0])
            #MSGName+='_'
            #MSGName+=str(job[1])
            f = open(SUBName, "w")
            f.write("executable = "+SHName)
            f.write("\n")
            #f.write("output ="+job[2]+"/HTCondor/MSG/"+MSGName+".out")
            #f.write("\n")
            #f.write("error ="+job[2]+"/HTCondor/MSG/"+MSGName+".err")
            #f.write("\n")
            #f.write("log ="+job[2]+"/HTCondor/MSG/"+MSGName+".log")
            #f.write("\n")
            f.write('requirements = (CERNEnvironment =!= "qa")')
            f.write("\n")
            f.write('+SoftUsed = "EDER-VIANN-E3"')
            f.write("\n")
            f.write('transfer_output_files = ""')
            f.write("\n")
            f.write('+JobFlavour = "workday"')
            f.write("\n")
            f.write('queue 1')
            f.write("\n")
            f.close()
            TotalLine='python3 '+job[2]+'/Code/Utilities/E3_DecorateEvalSeeds_Sub.py '+OptionLine
            f = open(SHName, "w")
            f.write("#!/bin/bash")
            f.write("\n")
            f.write("set -ux")
            f.write("\n")
            f.write(TotalLine)
            f.write("\n")
            f.close()
            subprocess.call(['condor_submit',SUBName])
            print(TotalLine," has been successfully submitted")

def SubmitCreateFakeSeedsJobsCondor(job):
            SHName=job[5]+'/HTCondor/SH/SH_E5_'+str(job[0])+'_'+str(job[1])+'.sh'
            SUBName=job[5]+'/HTCondor/SUB/SUB_E5_'+str(job[0])+'_'+str(job[1])+'.sub'
            MSGName='MSG_E5_'
            OptionLine=(' --Set '+str(job[0]))
            OptionLine+=(" --Subset "+str(job[1]))
            OptionLine+=(" --EOS "+str(job[6]))
            OptionLine+=(" --AFS "+str(job[5]))
            OptionLine+=(" --PlateZ "+str(job[2]))
            OptionLine+=(" --MaxTracks "+str(job[4]))
            OptionLine+=(" --SI_7 "+str(job[3]))
            MSGName+=str(job[0])
            MSGName+='_'
            MSGName+=str(job[1])
            f = open(SUBName, "w")
            f.write("executable = "+SHName)
            f.write("\n")
            #f.write("output ="+job[5]+"/HTCondor/MSG/"+MSGName+".out")
            #f.write("\n")
            #f.write("error ="+job[5]+"/HTCondor/MSG/"+MSGName+".err")
            #f.write("\n")
            #f.write("log ="+job[5]+"/HTCondor/MSG/"+MSGName+".log")
            #f.write("\n")
            f.write('requirements = (CERNEnvironment =!= "qa")')
            f.write("\n")
            f.write('+SoftUsed = "EDER-VIANN-E5"')
            f.write("\n")
            f.write('transfer_output_files = ""')
            f.write("\n")
            f.write('+JobFlavour = "workday"')
            f.write("\n")
            f.write('queue 1')
            f.write("\n")
            f.close()
            TotalLine='python3 '+job[5]+'/Code/Utilities/E5_GenerateFakeSeeds_Sub.py '+OptionLine
            f = open(SHName, "w")
            f.write("#!/bin/bash")
            f.write("\n")
            f.write("set -ux")
            f.write("\n")
            f.write(TotalLine)
            f.write("\n")
            f.close()
            subprocess.call(['condor_submit',SUBName])
            print(TotalLine," has been successfully submitted")
def SubmitDecorateFakeSeedsJobsCondor(job):
            SHName=job[3]+'/HTCondor/SH/SH_E6_'+str(job[0])+'_'+str(job[1])+'_'+str(job[2])+'.sh'
            SUBName=job[3]+'/HTCondor/SUB/SUB_E6_'+str(job[0])+'_'+str(job[1])+'_'+str(job[2])+'.sub'
            #MSGName='MSG_FDS_'
            OptionLine=(' --Set '+str(job[0]))
            OptionLine+=(" --SubSet "+str(job[1]))
            OptionLine+=(" --Fraction "+str(job[2]))
            OptionLine+=(" --EOS "+str(job[4]))
            OptionLine+=(" --AFS "+str(job[3]))
            #MSGName+=str(job[0])
            #MSGName+='_'
            #MSGName+=str(job[1])
            #MSGName+='_'
            #MSGName+=str(job[2])
            f = open(SUBName, "w")
            f.write("executable = "+SHName)
            f.write("\n")
            #f.write("output ="+job[3]+"/HTCondor/MSG/"+MSGName+".out")
            #f.write("\n")
            #f.write("error ="+job[3]+"/HTCondor/MSG/"+MSGName+".err")
            #f.write("\n")
            #f.write("log ="+job[3]+"/HTCondor/MSG/"+MSGName+".log")
            #f.write("\n")
            f.write('requirements = (CERNEnvironment =!= "qa")')
            f.write("\n")
            f.write('+SoftUsed = "EDER-VIANN-E6"')
            f.write("\n")
            f.write('transfer_output_files = ""')
            f.write("\n")
            f.write('+JobFlavour = "workday"')
            f.write("\n")
            f.write('queue 1')
            f.write("\n")
            f.close()
            TotalLine='python3 '+job[3]+'/Code/Utilities/E6_DecorateFakeSeeds_Sub.py '+OptionLine
            f = open(SHName, "w")
            f.write("#!/bin/bash")
            f.write("\n")
            f.write("set -ux")
            f.write("\n")
            f.write(TotalLine)
            f.write("\n")
            f.close()
            subprocess.call(['condor_submit',SUBName])
            print(TotalLine," has been successfully submitted")
def SubmitCreateKalmanSeedsJobsCondor(job):
            SHName=job[2]+'/HTCondor/SH/SH_E8_'+str(job[0])+'.sh'
            SUBName=job[2]+'/HTCondor/SUB/SUB_E8_'+str(job[0])+'.sub'
            MSGName='MSG_E8_'
            OptionLine=(" --Subset "+str(job[0]))
            OptionLine+=(" --EOS "+str(job[3]))
            OptionLine+=(" --AFS "+str(job[2]))
            OptionLine+=(" --MaxTracks "+str(job[1]))
            MSGName+=str(job[0])
            f = open(SUBName, "w")
            f.write("executable = "+SHName)
            f.write("\n")
            #f.write("output ="+job[2]+"/HTCondor/MSG/"+MSGName+".out")
            #f.write("\n")
            #f.write("error ="+job[2]+"/HTCondor/MSG/"+MSGName+".err")
            #f.write("\n")
            #f.write("log ="+job[2]+"/HTCondor/MSG/"+MSGName+".log")
            #f.write("\n")
            f.write('requirements = (CERNEnvironment =!= "qa")')
            f.write("\n")
            f.write('+SoftUsed = "EDER-VIANN-E8"')
            f.write("\n")
            f.write('transfer_output_files = ""')
            f.write("\n")
            f.write('+JobFlavour = "workday"')
            f.write("\n")
            f.write('queue 1')
            f.write("\n")
            f.close()
            TotalLine='python3 '+job[2]+'/Code/Utilities/E8_GenerateKalmanSeeds_Sub.py '+OptionLine
            f = open(SHName, "w")
            f.write("#!/bin/bash")
            f.write("\n")
            f.write("set -ux")
            f.write("\n")
            f.write(TotalLine)
            f.write("\n")
            f.close()
            subprocess.call(['condor_submit',SUBName])
            print(TotalLine," has been successfully submitted")
def SubmitDecorateKalmanSeedsJobsCondor(job):
            SHName=job[2]+'/HTCondor/SH/SH_E9_'+str(job[0])+'_'+str(job[1])+'.sh'
            SUBName=job[2]+'/HTCondor/SUB/SUB_E9_'+str(job[0])+'_'+str(job[1])+'.sub'
            #MSGName='MSG_DC_'
            OptionLine=(' --SubSet '+str(job[0]))
            OptionLine+=(" --Fraction "+str(job[1]))
            OptionLine+=(" --EOS "+str(job[3]))
            OptionLine+=(" --AFS "+str(job[2]))
            #MSGName+=str(job[0])
            #MSGName+='_'
            #MSGName+=str(job[1])
            f = open(SUBName, "w")
            f.write("executable = "+SHName)
            f.write("\n")
            #f.write("output ="+job[2]+"/HTCondor/MSG/"+MSGName+".out")
            #f.write("\n")
            #f.write("error ="+job[2]+"/HTCondor/MSG/"+MSGName+".err")
            #f.write("\n")
            #f.write("log ="+job[2]+"/HTCondor/MSG/"+MSGName+".log")
            #f.write("\n")
            f.write('requirements = (CERNEnvironment =!= "qa")')
            f.write("\n")
            f.write('+SoftUsed = "EDER-VIANN-E9"')
            f.write("\n")
            f.write('transfer_output_files = ""')
            f.write("\n")
            f.write('+JobFlavour = "workday"')
            f.write("\n")
            f.write('queue 1')
            f.write("\n")
            f.close()
            TotalLine='python3 '+job[2]+'/Code/Utilities/E9_DecorateKalmanSeeds_Sub.py '+OptionLine
            f = open(SHName, "w")
            f.write("#!/bin/bash")
            f.write("\n")
            f.write("set -ux")
            f.write("\n")
            f.write(TotalLine)
            f.write("\n")
            f.close()
            subprocess.call(['condor_submit',SUBName])
            print(TotalLine," has been successfully submitted")
#Training
def BSubmitCreateTrainSeedsJobsCondor(job):
            SHName=job[11]+'/HTCondor/SH/SH_M2_'+str(job[0])+'_'+str(job[1])+'.sh'
            SUBName=job[11]+'/HTCondor/SUB/SUB_M2_'+str(job[0])+'_'+str(job[1])+'.sub'
            #MSGName='MSG_M2_'
            OptionLine=(' --Set '+str(job[0]))
            OptionLine+=" --Subset $1"
            OptionLine+=(" --EOS "+str(job[12]))
            OptionLine+=(" --AFS "+str(job[11]))
            OptionLine+=(" --PlateZ "+str(job[2]))
            OptionLine+=(" --MaxTracks "+str(job[10]))
            OptionLine+=(" --SI_1 "+str(job[3]))
            OptionLine+=(" --SI_2 "+str(job[4]))
            OptionLine+=(" --SI_3 "+str(job[5]))
            OptionLine+=(" --SI_4 "+str(job[6]))
            OptionLine+=(" --SI_5 "+str(job[7]))
            OptionLine+=(" --SI_6 "+str(job[8]))
            OptionLine+=(" --SI_7 "+str(job[9]))
            OptionLine+=(" --NV "+str(job[13]))
            #MSGName+=str(job[0])
            #MSGName+='_'
            #MSGName+=str(job[1])
            f = open(SUBName, "w")
            f.write("executable = "+SHName)
            f.write("\n")
            #f.write("output ="+job[11]+"/HTCondor/MSG/"+MSGName+".out")
            #f.write("\n")
            #f.write("error ="+job[11]+"/HTCondor/MSG/"+MSGName+".err")
            #f.write("\n")
            #f.write("log ="+job[11]+"/HTCondor/MSG/"+MSGName+".log")
            #f.write("\n")
            f.write('requirements = (CERNEnvironment =!= "qa")')
            f.write("\n")
            f.write('arguments = $(Process)')
            f.write("\n")
            f.write('+SoftUsed = "EDER-VIANN-M2"')
            f.write("\n")
            f.write('transfer_output_files = ""')
            f.write("\n")
            f.write('+JobFlavour = "workday"')
            f.write("\n")
            f.write('queue '+str(job[1]))
            f.write("\n")
            f.close()
            TotalLine='python3 '+job[11]+'/Code/Utilities/M2_GenerateTrainSeeds_Sub.py '+OptionLine
            f = open(SHName, "w")
            f.write("#!/bin/bash")
            f.write("\n")
            f.write("set -ux")
            f.write("\n")
            f.write(TotalLine)
            f.write("\n")
            f.close()
            subprocess.call(['condor_submit',SUBName])
            print(TotalLine," has been successfully submitted")
def SubmitCreateTrainSeedsJobsCondor(job):
            SHName=job[11]+'/HTCondor/SH/SH_M2_'+str(job[0])+'_'+str(job[1])+'.sh'
            SUBName=job[11]+'/HTCondor/SUB/SUB_M2_'+str(job[0])+'_'+str(job[1])+'.sub'
            #MSGName='MSG_M2_'
            OptionLine=(' --Set '+str(job[0]))
            OptionLine+=(" --Subset "+str(job[1]))
            OptionLine+=(" --EOS "+str(job[12]))
            OptionLine+=(" --AFS "+str(job[11]))
            OptionLine+=(" --PlateZ "+str(job[2]))
            OptionLine+=(" --MaxTracks "+str(job[10]))
            OptionLine+=(" --SI_1 "+str(job[3]))
            OptionLine+=(" --SI_2 "+str(job[4]))
            OptionLine+=(" --SI_3 "+str(job[5]))
            OptionLine+=(" --SI_4 "+str(job[6]))
            OptionLine+=(" --SI_5 "+str(job[7]))
            OptionLine+=(" --SI_6 "+str(job[8]))
            OptionLine+=(" --SI_7 "+str(job[9]))
            OptionLine+=(" --NV "+str(job[13]))
            #MSGName+=str(job[0])
            #MSGName+='_'
            #MSGName+=str(job[1])
            f = open(SUBName, "w")
            f.write("executable = "+SHName)
            f.write("\n")
            #f.write("output ="+job[11]+"/HTCondor/MSG/"+MSGName+".out")
            #f.write("\n")
            #f.write("error ="+job[11]+"/HTCondor/MSG/"+MSGName+".err")
            #f.write("\n")
            #f.write("log ="+job[11]+"/HTCondor/MSG/"+MSGName+".log")
            #f.write("\n")
            f.write('requirements = (CERNEnvironment =!= "qa")')
            f.write("\n")
            f.write('+SoftUsed = "EDER-VIANN-M2"')
            f.write("\n")
            f.write('transfer_output_files = ""')
            f.write("\n")
            f.write('+JobFlavour = "workday"')
            f.write("\n")
            f.write('queue 1')
            f.write("\n")
            f.close()
            TotalLine='python3 '+job[11]+'/Code/Utilities/M2_GenerateTrainSeeds_Sub.py '+OptionLine
            f = open(SHName, "w")
            f.write("#!/bin/bash")
            f.write("\n")
            f.write("set -ux")
            f.write("\n")
            f.write(TotalLine)
            f.write("\n")
            f.close()
            subprocess.call(['condor_submit',SUBName])
            print(TotalLine," has been successfully submitted")

def SubmitTrainJobCondor(AFS_DIR,EOS_DIR,job_params,mode):
            SHName=AFS_DIR+'/HTCondor/SH/SH_M5_'+str(job_params[0])+'.sh'
            SUBName=AFS_DIR+'/HTCondor/SUB/SUB_M5_'+str(job_params[0])+'.sub'
            MSGName='MSG_M5_'
            if mode=='New':
                OptionLine=' --Mode Create'
            else:
                OptionLine=' --Mode Train'
            OptionLine+=(" --ImageSet "+str(job_params[0]))
            OptionLine+=(' --Epoch '+str(job_params[1]))
            OptionLine+=(' --DNA '+'"'+str(job_params[2])+'"')
            OptionLine+=(" --EOS "+EOS_DIR)
            OptionLine+=(" --AFS "+AFS_DIR)
            OptionLine+=(" --LR "+str(job_params[3]))
            OptionLine+=(" --ModelName "+str(job_params[4]))
            OptionLine+=(" --ModelNewName "+str(job_params[5]))
            MSGName+=(str(job_params[0]))
            f = open(SUBName, "w")
            f.write("executable = "+SHName)
            f.write("\n")
            f.write("output ="+AFS_DIR+"/HTCondor/MSG/"+MSGName+".out")
            f.write("\n")
            f.write("error ="+AFS_DIR+"/HTCondor/MSG/"+MSGName+".err")
            f.write("\n")
            f.write("log ="+AFS_DIR+"/HTCondor/MSG/"+MSGName+".log")
            f.write("\n")
            f.write('requirements = (CERNEnvironment =!= "qa")')
            f.write("\n")
            f.write('request_gpus = 1')
            f.write("\n")
            f.write('+SoftUsed = "EDER-VIANN-M5"')
            f.write("\n")
            f.write('transfer_output_files = ""')
            f.write("\n")
            f.write('+JobFlavour = "workday"')
            f.write("\n")
            f.write('queue 1')
            f.write("\n")
            f.close()
            TotalLine='python3 '+AFS_DIR+'/Code/Utilities/M5_TrainModel_Sub.py '+OptionLine
            f = open(SHName, "w")
            f.write("#!/bin/bash")
            f.write("\n")
            f.write("set -ux")
            f.write("\n")
            f.write(TotalLine)
            f.write("\n")
            f.close()
            subprocess.call(['condor_submit',SUBName])
            print(TotalLine," has been successfully submitted")
def BSubmitImageJobsCondor(job):
            SHName=job[7]+'/HTCondor/SH/SH_M3_'+str(job[0])+'_'+str(job[1])+'_'+str(job[2])+'.sh'
            SUBName=job[7]+'/HTCondor/SUB/SUB_M3_'+str(job[0])+'_'+str(job[1])+'_'+str(job[2])+'.sub'
            #MSGName='MSG_M3_'
            OptionLine=(' --Set '+str(job[0]))
            OptionLine+=(" --SubSet "+str(job[1]))
            OptionLine+=" --Fraction $1"
            OptionLine+=(" --EOS "+str(job[8]))
            OptionLine+=(" --AFS "+str(job[7]))
            OptionLine+=(" --VO_max_Z "+str(job[3]))
            OptionLine+=(" --VO_min_Z "+str(job[4]))
            OptionLine+=(" --VO_T "+str(job[5]))
            OptionLine+=(" --MaxDoca "+str(job[6]))
            #MSGName+=str(job[0])
            #MSGName+='_'
            #MSGName+=str(job[1])
            #MSGName+='_'
            #MSGName+=str(job[2])
            f = open(SUBName, "w")
            f.write("executable = "+SHName)
            f.write("\n")
            #f.write("output ="+job[11]+"/HTCondor/MSG/"+MSGName+".out")
            #f.write("\n")
            #f.write("error ="+job[11]+"/HTCondor/MSG/"+MSGName+".err")
            #f.write("\n")
            #f.write("log ="+job[11]+"/HTCondor/MSG/"+MSGName+".log")
            #f.write("\n")
            f.write('requirements = (CERNEnvironment =!= "qa")')
            f.write("\n")
            f.write('arguments = $(Process)')
            f.write("\n")
            f.write('+SoftUsed = "EDER-VIANN-M3"')
            f.write("\n")
            f.write('transfer_output_files = ""')
            f.write("\n")
            f.write('+JobFlavour = "workday"')
            f.write("\n")
            f.write('queue '+str(job[2]))
            f.write("\n")
            f.close()
            TotalLine='python3 '+job[7]+'/Code/Utilities/M3_GenerateImages_Sub.py '+OptionLine
            f = open(SHName, "w")
            f.write("#!/bin/bash")
            f.write("\n")
            f.write("set -ux")
            f.write("\n")
            f.write(TotalLine)
            f.write("\n")
            f.close()
            subprocess.call(['condor_submit',SUBName])
            print(TotalLine," has been successfully submitted")
def SubmitImageJobsCondor(job):
            SHName=job[7]+'/HTCondor/SH/SH_M3_'+str(job[0])+'_'+str(job[1])+'_'+str(job[2])+'.sh'
            SUBName=job[7]+'/HTCondor/SUB/SUB_M3_'+str(job[0])+'_'+str(job[1])+'_'+str(job[2])+'.sub'
            #MSGName='MSG_M3_'
            OptionLine=(' --Set '+str(job[0]))
            OptionLine+=(" --SubSet "+str(job[1]))
            OptionLine+=(" --Fraction "+str(job[2]))
            OptionLine+=(" --EOS "+str(job[8]))
            OptionLine+=(" --AFS "+str(job[7]))
            OptionLine+=(" --VO_max_Z "+str(job[3]))
            OptionLine+=(" --VO_min_Z "+str(job[4]))
            OptionLine+=(" --VO_T "+str(job[5]))
            OptionLine+=(" --MaxDoca "+str(job[6]))
            #MSGName+=str(job[0])
            #MSGName+='_'
            #MSGName+=str(job[1])
            #MSGName+='_'
            #MSGName+=str(job[2])
            f = open(SUBName, "w")
            f.write("executable = "+SHName)
            f.write("\n")
            #f.write("output ="+job[7]+"/HTCondor/MSG/"+MSGName+".out")
            #f.write("\n")
            #f.write("error ="+job[7]+"/HTCondor/MSG/"+MSGName+".err")
            #f.write("\n")
            #f.write("log ="+job[7]+"/HTCondor/MSG/"+MSGName+".log")
            #f.write("\n")
            f.write('requirements = (CERNEnvironment =!= "qa")')
            f.write("\n")
            f.write('+SoftUsed = "EDER-VIANN-M3"')
            f.write("\n")
            f.write('transfer_output_files = ""')
            f.write("\n")
            f.write('+JobFlavour = "workday"')
            f.write("\n")
            f.write('queue 1')
            f.write("\n")
            f.close()
            TotalLine='python3 '+job[7]+'/Code/Utilities/M3_GenerateImages_Sub.py '+OptionLine
            f = open(SHName, "w")
            f.write("#!/bin/bash")
            f.write("\n")
            f.write("set -ux")
            f.write("\n")
            f.write(TotalLine)
            f.write("\n")
            f.close()
            subprocess.call(['condor_submit',SUBName])
            print(TotalLine," has been successfully submitted")

def BSubmitRenderImageJobsCondor(job):
       if job[0]=='Val':
            SHName=job[6]+'/HTCondor/SH/SH_M4_val.sh'
            SUBName=job[6]+'/HTCondor/SUB/SUB_M4_val.sub'
            #MSGName='MSG_M4_val_'
       else:
            SHName=job[6]+'/HTCondor/SH/SH_M4_Train.sh'
            SUBName=job[6]+'/HTCondor/SUB/SUB_M4_Train.sub'
            #MSGName='MSG_M4_val_'
       OptionLine=(" --Fraction $1")
       OptionLine+=(" --EOS "+str(job[7]))
       OptionLine+=(" --SetType "+str(job[0]))
       OptionLine+=(" --AFS "+str(job[6]))
       OptionLine+=(" --resolution "+str(job[2]))
       OptionLine+=(" --MaxX "+str(job[3]))
       OptionLine+=(" --MaxY "+str(job[4]))
       OptionLine+=(" --MaxZ "+str(job[5]))
       #MSGName+=str(job[0])
       #MSGName+='_'
       #MSGName+=str(job[1])
       f = open(SUBName, "w")
       f.write("executable = "+SHName)
       f.write("\n")
       #f.write("output ="+job[7]+"/HTCondor/MSG/"+MSGName+".out")
       #f.write("\n")
       #f.write("error ="+job[7]+"/HTCondor/MSG/"+MSGName+".err")
       #f.write("\n")
       #f.write("log ="+job[7]+"/HTCondor/MSG/"+MSGName+".log")
       #f.write("\n")
       f.write('requirements = (CERNEnvironment =!= "qa")')
       f.write("\n")
       f.write('arguments = $(Process)')
       f.write("\n")
       f.write('+SoftUsed = "EDER-VIANN-M4"')
       f.write("\n")
       f.write('transfer_output_files = ""')
       f.write("\n")
       f.write('+JobFlavour = "workday"')
       f.write("\n")
       f.write('queue '+str(job[1]))
       f.write("\n")
       f.close()
       TotalLine='python3 '+job[6]+'/Code/Utilities/M4_RenderImages_Sub.py '+OptionLine
       f = open(SHName, "w")
       f.write("#!/bin/bash")
       f.write("\n")
       f.write("set -ux")
       f.write("\n")
       f.write(TotalLine)
       f.write("\n")
       f.close()
       subprocess.call(['condor_submit',SUBName])
       print(TotalLine," has been successfully submitted")

def SubmitRenderImageJobsCondor(job):
       if job[0]=='Val':
            SHName=job[6]+'/HTCondor/SH/SH_M4_val.sh'
            SUBName=job[6]+'/HTCondor/SUB/SUB_M4_val.sub'
            MSGName='MSG_M4_val_'
       else:
            SHName=job[6]+'/HTCondor/SH/SH_M4_Train'+str(job[1])+'.sh'
            SUBName=job[6]+'/HTCondor/SUB/SUB_M4_Train'+str(job[1])+'.sub'
            MSGName='MSG_M4_val_'
       OptionLine=(" --Fraction "+ str(job[1]))
       OptionLine+=(" --SetType "+str(job[0]))
       OptionLine+=(" --EOS "+str(job[7]))
       OptionLine+=(" --AFS "+str(job[6]))
       OptionLine+=(" --resolution "+str(job[2]))
       OptionLine+=(" --MaxX "+str(job[3]))
       OptionLine+=(" --MaxY "+str(job[4]))
       OptionLine+=(" --MaxZ "+str(job[5]))
       MSGName+=str(job[0])
       MSGName+='_'
       MSGName+=str(job[1])
       f = open(SUBName, "w")
       f.write("executable = "+SHName)
       f.write("\n")
       f.write("output ="+job[6]+"/HTCondor/MSG/"+MSGName+".out")
       f.write("\n")
       f.write("error ="+job[6]+"/HTCondor/MSG/"+MSGName+".err")
       f.write("\n")
       f.write("log ="+job[6]+"/HTCondor/MSG/"+MSGName+".log")
       f.write("\n")
       f.write('requirements = (CERNEnvironment =!= "qa")')
       f.write("\n")
       f.write('+SoftUsed = "EDER-VIANN-M4"')
       f.write("\n")
       f.write('transfer_output_files = ""')
       f.write("\n")
       f.write('+JobFlavour = "workday"')
       f.write("\n")
       f.write('queue 1')
       f.write("\n")
       f.close()
       TotalLine='python3 '+job[6]+'/Code/Utilities/M4_RenderImages_Sub.py '+OptionLine
       f = open(SHName, "w")
       f.write("#!/bin/bash")
       f.write("\n")
       f.write("set -ux")
       f.write("\n")
       f.write(TotalLine)
       f.write("\n")
       f.close()
       subprocess.call(['condor_submit',SUBName])
       print(TotalLine," has been successfully submitted")

def   RecCleanUp(AFS_DIR, EOS_DIR, Process, FileNames, ProcessId):
      subprocess.call(['condor_rm', '-constraint', ProcessId])
      EOSsubDIR=EOS_DIR+'/'+'EDER-VIANN'
      EOSsubModelDIR=EOSsubDIR+'/'+'Data/REC_SET'
      folder =  EOSsubModelDIR
      for f in FileNames:
          CleanFolder(folder,f)
      folder =  AFS_DIR+'/HTCondor/SH'
      CleanFolder(folder,'SH_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/SUB'
      CleanFolder(folder,'SUB_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/MSG'
      CleanFolder(folder,'MSG_'+Process+'_')

def   EvalCleanUp(AFS_DIR, EOS_DIR, Process, FileNames, ProcessId):
      subprocess.call(['condor_rm', '-constraint', ProcessId])
      EOSsubDIR=EOS_DIR+'/'+'EDER-VIANN'
      EOSsubModelDIR=EOSsubDIR+'/'+'Data/TEST_SET'
      folder =  EOSsubModelDIR
      for f in FileNames:
          CleanFolder(folder,f)
      folder =  AFS_DIR+'/HTCondor/SH'
      CleanFolder(folder,'SH_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/SUB'
      CleanFolder(folder,'SUB_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/MSG'
      CleanFolder(folder,'MSG_'+Process+'_')

def   TrainCleanUp(AFS_DIR, EOS_DIR, Process, FileNames, ProcessId):
      subprocess.call(['condor_rm', '-constraint', ProcessId])
      EOSsubDIR=EOS_DIR+'/'+'EDER-VIANN'
      EOSsubModelDIR=EOSsubDIR+'/'+'Data/TRAIN_SET'
      folder =  EOSsubModelDIR
      for f in FileNames:
          CleanFolder(folder,f)
      EOSsubModelDIR=EOSsubDIR+'/'+'Models'
      folder =  EOSsubModelDIR
      for f in FileNames:
          CleanFolder(folder,f)
      folder =  AFS_DIR+'/HTCondor/SH'
      CleanFolder(folder,'SH_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/SUB'
      CleanFolder(folder,'SUB_'+Process+'_')
      folder =  AFS_DIR+'/HTCondor/MSG'
      CleanFolder(folder,'MSG_'+Process+'_')

def LoadRenderImages(Seeds,StartSeed,EndSeed):
    import tensorflow as tf
    from tensorflow import keras
    NewSeeds=Seeds[StartSeed-1:min(EndSeed,len(Seeds))]
    ImagesY=np.empty([len(NewSeeds),1])
    ImagesX=np.empty([len(NewSeeds),NewSeeds[0].H,NewSeeds[0].W,NewSeeds[0].L],dtype=np.float16)
    for im in range(len(NewSeeds)):
        if hasattr(NewSeeds[im],'MC_truth_label'):
           ImagesY[im]=int(float(NewSeeds[im].MC_truth_label))
        else:
           ImagesY[im]=0
        BlankRenderedImage=[]
        for x in range(-NewSeeds[im].bX,NewSeeds[im].bX):
          for y in range(-NewSeeds[im].bY,NewSeeds[im].bY):
            for z in range(0,NewSeeds[im].bZ):
             BlankRenderedImage.append(0.0)
        RenderedImage = np.array(BlankRenderedImage)
        RenderedImage = np.reshape(RenderedImage,(NewSeeds[im].H,NewSeeds[im].W,NewSeeds[im].L))
        for Hits in NewSeeds[im].TrackPrint:
                if abs(Hits[0])<NewSeeds[im].bX and abs(Hits[1])<NewSeeds[im].bY and abs(Hits[2])<NewSeeds[im].bZ:
                   RenderedImage[Hits[0]+NewSeeds[im].bX][Hits[1]+NewSeeds[im].bY][Hits[2]]=1.0
        ImagesX[im]=RenderedImage
    ImagesX= ImagesX[..., np.newaxis]
    ImagesY=tf.keras.utils.to_categorical(ImagesY,2)
    return (ImagesX,ImagesY)
