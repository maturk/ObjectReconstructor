'''
vis script

Control:
key q: quit
key Q: sys.exit(0)
key n: zoom in
key m: zoom out
key s: save screenshot to 'show3d.png'
Mouse: rotate

Additional Parameters
---------------------
normalizecolor:
	if True (default), scale the maximum color to 1 for each channel.
magnifyBlue:
	if True, magnify the blue dots to make them more visible
background:
	the background color. Defaults to black (0,0,0)
freezerot:
	disable rotation
'''


import numpy as np
import cv2
import sys
showsz=512
mousex,mousey=0.5,0.5
zoom=1.0
changed=True
def onmouse(*args):
	global mousex,mousey,changed
	y=args[1]
	x=args[2]
	mousex=x/float(showsz)
	mousey=y/float(showsz)
	changed=True

def showpoints(xyz,c0=None,c1=None,c2=None,waittime=0,showrot=False,magnifyBlue=0,freezerot=False,background=(0,0,0),normalizecolor=True):
	global showsz,mousex,mousey,zoom,changed
	cv2.namedWindow('show3d')
	cv2.moveWindow('show3d',0,0)
	cv2.setMouseCallback('show3d',onmouse)
	if len(xyz.shape)!=2 or xyz.shape[1]!=3:
		raise Exception('showpoints expects (n,3) shape for xyz')
	if c0 is not None and c0.shape!=xyz.shape[:1]:
		raise Exception('showpoints expects (n,) shape for c0')
	if c1 is not None and c1.shape!=xyz.shape[:1]:
		raise Exception('showpoints expects (n,) shape for c1')
	if c2 is not None and c2.shape!=xyz.shape[:1]:
		raise Exception('showpoints expects (n,) shape for c2')
	xyz=xyz-xyz.mean(axis=0)
	radius=((xyz**2).sum(axis=-1)**0.5).max()
	xyz/=(radius*2.2)/showsz
	if c0 is None:
		c0=np.zeros((len(xyz),),dtype='float32')+255
	if c1 is None:
		c1=c0
	if c2 is None:
		c2=c0
	if normalizecolor:
		c0=c0/((c0.max()+1e-14)/255.0)
		c1=c1/((c1.max()+1e-14)/255.0)
		c2=c2/((c2.max()+1e-14)/255.0)

	show=np.zeros((showsz,showsz,3),dtype='uint8')
	def render():
		rotmat=np.eye(3)
		if not freezerot:
			xangle=(mousey-0.5)*np.pi*1.2
		else:
			xangle=0
		rotmat=rotmat.dot(np.array([
			[1.0,0.0,0.0],
			[0.0,np.cos(xangle),-np.sin(xangle)],
			[0.0,np.sin(xangle),np.cos(xangle)],
			]))
		if not freezerot:
			yangle=(mousex-0.5)*np.pi*1.2
		else:
			yangle=0
		rotmat=rotmat.dot(np.array([
			[np.cos(yangle),0.0,-np.sin(yangle)],
			[0.0,1.0,0.0],
			[np.sin(yangle),0.0,np.cos(yangle)],
			]))
		rotmat*=zoom
		nxyz=xyz.dot(rotmat)
		nz=nxyz[:,2].argsort()
		nxyz=nxyz[nz]
		nxyz=(nxyz[:,:2]+[showsz/2,showsz/2]).astype('int32')
		p=nxyz[:,0]*showsz+nxyz[:,1]
		show[:]=background
		m=(nxyz[:,0]>=0)*(nxyz[:,0]<showsz)*(nxyz[:,1]>=0)*(nxyz[:,1]<showsz)
		show.reshape((showsz*showsz,3))[p[m],1]=c0[nz][m]
		show.reshape((showsz*showsz,3))[p[m],2]=c1[nz][m]
		show.reshape((showsz*showsz,3))[p[m],0]=c2[nz][m]
		if magnifyBlue>0:
			show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],1,axis=0))
			if magnifyBlue>=2:
				show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],-1,axis=0))
			show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],1,axis=1))
			if magnifyBlue>=2:
				show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],-1,axis=1))
		if showrot:
			cv2.putText(show,'xangle %d'%(int(xangle/np.pi*180)),(30,showsz-30),0,0.5,cv2.cv.CV_RGB(255,0,0))
			cv2.putText(show,'yangle %d'%(int(yangle/np.pi*180)),(30,showsz-50),0,0.5,cv2.cv.CV_RGB(255,0,0))
			cv2.putText(show,'zoom %d%%'%(int(zoom*100)),(30,showsz-70),0,0.5,cv2.cv.CV_RGB(255,0,0))
	changed=True
	while True:
		if changed:
			render()
			changed=False
		cv2.imshow('show3d',show)
		if waittime==0:
			cmd=cv2.waitKey(10)%256
		else:
			cmd=cv2.waitKey(waittime)%256
		if cmd==ord('q'):
			break
		elif cmd==ord('Q'):
			sys.exit(0)
		if cmd==ord('n'):
			zoom*=1.1
			changed=True
		elif cmd==ord('m'):
			zoom/=1.1
			changed=True
		elif cmd==ord('r'):
			zoom=1.0
			changed=True
		elif cmd==ord('s'):
			cv2.imwrite('show3d.png',show)
		if waittime!=0:
			break
	return cmd

def renderpoints(xyz,path, zangle = 0, c0=None,c1=None,c2=None,waittime=0,showrot=False,magnifyBlue=0,freezerot=True,background=(0,0,0),normalizecolor=True):
	global showsz,mousex,mousey,zoom,changed
	if len(xyz.shape)!=2 or xyz.shape[1]!=3:
		raise Exception('showpoints expects (n,3) shape for xyz')
	if c0 is not None and c0.shape!=xyz.shape[:1]:
		raise Exception('showpoints expects (n,) shape for c0')
	if c1 is not None and c1.shape!=xyz.shape[:1]:
		raise Exception('showpoints expects (n,) shape for c1')
	if c2 is not None and c2.shape!=xyz.shape[:1]:
		raise Exception('showpoints expects (n,) shape for c2')
	xyz=xyz-xyz.mean(axis=0)
	radius=((xyz**2).sum(axis=-1)**0.5).max() 
	xyz/=(radius*2.2)/showsz
	if c0 is None:
		c0=np.zeros((len(xyz),),dtype='float32')+255
	if c1 is None:
		c1=c0
	if c2 is None:
		c2=c0
	if normalizecolor:
		c0=c0/((c0.max()+1e-14)/255.0)
		c1=c1/((c1.max()+1e-14)/255.0)
		c2=c2/((c2.max()+1e-14)/255.0)

	show=np.zeros((showsz,showsz,3),dtype='uint8')
	def render(zangle = 0):
		zangle = zangle*0.0174533 # degrees to radians
		rotmat=np.eye(3)
		if not freezerot:
			xangle=(mousey-0.5)*np.pi*1.2
		else:
			xangle=1.57
		rotmat=rotmat.dot(np.array([
			[1.0,0.0,0.0],
			[0.0,np.cos(xangle),-np.sin(xangle)],
			[0.0,np.sin(xangle),np.cos(xangle)],
			]))
		if zangle:
			rotmat=rotmat.dot(np.array([
				[np.cos(zangle),np.sin(zangle), 0.0],
				[np.sin(zangle),np.cos(zangle), 0.0],
				[0.0, 0.0, 1.0],
				]))
		if not freezerot:
			yangle=(mousex-0.5)*np.pi*1.2
		else:
			yangle= 1.57 -0.5
		rotmat=rotmat.dot(np.array([
			[np.cos(yangle),0.0,-np.sin(yangle)],
			[0.0,1.0,0.0],
			[np.sin(yangle),0.0,np.cos(yangle)],
			]))
			
		rotmat*=zoom
		nxyz=xyz.dot(rotmat)
		nz=nxyz[:,2].argsort()
		nxyz=nxyz[nz]
		nxyz=(nxyz[:,:2]+[showsz/2,showsz/2]).astype('int32')
		p=nxyz[:,0]*showsz+nxyz[:,1]
		show[:]=background
		m=(nxyz[:,0]>=0)*(nxyz[:,0]<showsz)*(nxyz[:,1]>=0)*(nxyz[:,1]<showsz)
		show.reshape((showsz*showsz,3))[p[m],1]=c0[nz][m]
		show.reshape((showsz*showsz,3))[p[m],2]=c1[nz][m]
		show.reshape((showsz*showsz,3))[p[m],0]=c2[nz][m]
		if magnifyBlue>0:
			show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],1,axis=0))
			if magnifyBlue>=2:
				show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],-1,axis=0))
			show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],1,axis=1))
			if magnifyBlue>=2:
				show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],-1,axis=1))
	render(zangle = zangle)
	cv2.imwrite(path,show)
 
if __name__=='__main__':
	showpoints(np.random.rand(10000,3))
	green=np.linspace(0,1,10000)
	red=np.linspace(1,0,10000)**0.5
	blue=np.linspace(1,0,10000)
 	# Show gui
	showpoints(np.random.rand(10000,3),green,red,blue,magnifyBlue=True)
	# Save render
 	# path = ''
  	# zangle = 90	
	# renderpoints(np.random.rand(10000,3),path = path, zangle = zangle, green,red,blue,magnifyBlue=True)	
 