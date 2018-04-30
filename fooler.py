# Neural Photo Editor
# A Brock, 2016

# Imports

# Note that I dislike the * on the Tkinter import, but all the tutorials seem to do that so I stuck with it.
from tkinter import *
# This produces an OS-dependent color selector. I like the windows one best, and can't stand the linux one.
from tkinter.colorchooser import askcolor
from collections import OrderedDict
from PIL import Image, ImageTk
import numpy as np
import scipy.misc
import tensorflow as tf
import cgan as cgan
from scipy.misc import imsave
from matplotlib import pyplot as plt
from generator  import generator


# from API import IAN

class fooler():
    def __init__(self, **kwargs):
        self.model = kwargs['model']
        self.log_dir = kwargs['log_dir']
        self.sess = tf.InteractiveSession(graph=self.model.orig.graph)
        self.X = np.zeros((64, 64, 3), dtype='float32')

        # self.sess.run(tf.global_variables_initializer())
        # tf.reset_default_graph()
        # self.saver = tf.train.import_meta_graph(self.log_dir + self.meta_graph)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(
            self.sess, tf.train.latest_checkpoint(self.log_dir))

        # summary_writer = tf.summary.FileWriter('./', self.sess.graph)
        # summary_writer.flush()

    def updateX(self, X):
        self.X = X

    def sample_at(self, z):
        image = self.sess.run(self.model.G_samp,
                              feed_dict={self.model.Z: z,
                                         self.model.TRAIN: False,
                                         self.model.EPS_BS: 1})
        print(image.shape)
        # imsave('trial.png', image[0])

        return image.swapaxes(1, 3).swapaxes(2, 3)
        # return image

    def imgradRGB(self, c1, r1, c2, r2, RGB, z):
        x_hat = self.model.G_orig
        # x_hat = generator(self.model.mean, reuse=True)
        # z_t = self.model.latent
        RGB = RGB.swapaxes(1, 3).swapaxes(1, 2)
        # print(RGB.shape)
        RGB_t = tf.convert_to_tensor(RGB)
        # z_t = tf.convert_to_tensor(z)
        expr = tf.reduce_mean(
            tf.square(x_hat[0, r1:r2, c1:c2, :] - RGB_t[0, r1:r2, c1:c2, :]))

        # grad = tf.gradients([expr], [z_t])
        grad = tf.gradients([expr], [self.model.latent])

        grad_out, loss   = self.sess.run(
            [grad, expr], feed_dict={self.model.orig: self.X,
                             self.model.Z: z,
                             self.model.EPS_BS: 1,
                             self.model.TRAIN: False})

        return grad_out

    def encode_images(self, x):
        # print(x.shape)
        self.X = x.swapaxes(1, 3).swapaxes(1, 2)
        # print(self.X.shape)
        return self.sess.run(self.model.latent, feed_dict={self.model.orig: self.X,
                                                           self.model.EPS_BS: 1,
                                                           self.model.TRAIN: False})

    def get_zdim(self):
        return self.model.lat_size

# Step 1: Create theano functions


# Initialize model
# model = IAN(config_path = 'IAN_simple.py', dnn = True)
net = cgan.CGAN()
model = fooler(model=net, log_dir='results/trial/LOGS')


# z_trial = np.eye(100)[45].reshape(-1, 100)
# out = model.sample_at(z_trial)
# imsave('trial.png', out[0])
# imsave('trial.png',out.reshape(5,5,64,64,3).swapaxes(1,2).reshape(5*64,-1,3))


# z_trial = np.random.uniform(-1.,1., size=(25,100))
# out = model.sample_at(z_trial)
# print(out.shape)
# imsave('trial2.png',out.reshape(5,5,64,64,3).swapaxes(1,2).reshape(5*64,-1,3))

# import sys
# sys.exit()
# # Prepare GUI functions
# print('Compiling remaining functions')

# Create master
master = Tk()
master.title("Neural Photo Editor")

# RGB interpreter convenience function


def rgb(r, g, b):
    return '#%02x%02x%02x' % (int(r), int(g), int(b))

# Convert RGB to bi-directional RB scale.


def rb(i):
    # return rgb(int(i*int(i>0)),0, -int(i*int(i<0)))
    return rgb(255+max(int(i*int(i < 0)), -255), 255-min(abs(int(i)), 255), 255-min(int(i*int(i > 0)), 255))

# Convenience functions to go from [0,255] to [-1,1] and [-1,1] to [0,255]


def to_tanh(input):
    # print('max tanh')
    # print(np.max(input))
    return input/255.0
    # return 1.0*(input/255.0)


def from_tanh(input):
    return 255.0*input
    #return 255.0*(input)/1.0


# Ground truth image
GIM = np.asarray(np.load('CelebAValid2.npz')['arr_0'][420])

# Image for modification
IM = GIM

# Reconstruction
RECON = IM

# Error between reconstruction and current image
ERROR = np.zeros(np.shape(IM), dtype=np.float32)

# Change between Recon and Current
DELTA = np.zeros(np.shape(IM), dtype=np.float32)

# User-Painted Mask, currently not implemented.
USER_MASK = np.mean(DELTA, axis=0)

# Are we operating on a photo or a sample?
SAMPLE_FLAG = 0


# Latent Canvas Variables
# Latent Square dimensions
dim = [10, 10]

# Squared Latent Array
Z = np.random.randn(dim[0], dim[1])

# Pixel-wise resolution for latent canvas
res = 16

# Array that holds the actual latent canvas
r = np.zeros((res*dim[0], res*dim[1]), dtype=np.float32)

# Painted rectangles for free-form latent painting
painted_rects = []

# Actual latent rectangles
rects = np.zeros((dim[0], dim[1]), dtype=int)

# Output Display Variables

# RGB paintbrush array
myRGB = np.zeros((1, 3, 64, 64), dtype=np.float32)

# Canvas width and height
canvas_width = 800
canvas_height = 800

# border width
bd = 2
# Brush color
color = IntVar()
color.set(0)

# Brush size
d = IntVar()
d.set(12)

# Selected Color
mycol = (0, 0, 0)

from skimage.transform import resize


def update_photo(data=None, widget=None):
    """
    # Function to update display
    if you pass data, it will upsample and paint canvas using data
    else will use global Z to get a generated image and update canvas"""
    global Z
    # print(Z[:5,0])
    if data is None:  # By default, assume we're updating with the current value of Z
        data = np.repeat(np.repeat(np.uint8(
            from_tanh(model.sample_at(np.float32([Z.flatten()]))[0])), 4, 1), 4, 2)
        # data = model.sample_at(np.float32([Z.flatten()]))[0]

        # data = resize(data.swapaxes(0,2).swapaxes(0,1), (64*4,64*4))
        # data = np.repeat(np.repeat(np.uint8(
            # from_tanh(model.sample_at(np.float32([Z.flatten()]))[0])), 4, 1), 4, 2)
        # print('data shape')
        # print(data.shape)
    else:
        data = np.repeat(np.repeat(np.uint8(data), 4, 1), 4, 2)
        # data = model.sample_at(np.float32([Z.flatten()]))[0]
        # data = resize(data.swapaxes(0,2).swapaxes(0,1), (64*4,64*4))
        # print('data shape')
        # print(data.shape)

    if widget is None:
        widget = output
    # Reshape image to canvas
    mshape = (4*64, 4*64, 1)
    # print(data.shape)
    # print(data[:2, :2, :2])
    im = Image.fromarray(np.concatenate([np.reshape(data[0], mshape), np.reshape(
        data[1], mshape), np.reshape(data[2], mshape)], axis=2), mode='RGB')
    # im = Image.fromarray(data, mode='RGB')

    # Make sure photo is an object of the current widget so the garbage collector doesn't wreck it

    # plt.figure()
    # plt.imshow(im)
    # plt.savefig('temp.py')
    widget.photo = ImageTk.PhotoImage(image=im)
    widget.create_image(0, 0, image=widget.photo, anchor=NW)
    widget.tag_raise(pixel_rect)

# Function to update the latent canvas.


def update_canvas(widget=None):
    global r, Z, res, rects, painted_rects
    if widget is None:
        widget = w
    # Update display values
    r = np.repeat(
        np.repeat(Z, r.shape[0]//Z.shape[0], 0), r.shape[1]//Z.shape[1], 1)

    # If we're letting freeform painting happen, delete the painted rectangles
    for p in painted_rects:
        w.delete(p)
    painted_rects = []

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            w.itemconfig(int(rects[i, j]), fill=rb(
                255*Z[i, j]), outline=rb(255*Z[i, j]))



def move_mouse(event):
    """Function to move the paintbrush""" 
    global output
    # print('inside move')
    # using a rectanglee width equivalent to d/4 (so 1-16)

    # First, get location and extent of local patch
    x, y = event.x//4, event.y//4
    brush_width = ((d.get()//4)+1)

    # if x is near the left corner, then the minimum x is dependent on how close it is to the left
    # This 64 may need to change if the canvas size changes
    xmin = max(min(x-brush_width//2, 64 - brush_width), 0)
    xmax = xmin+brush_width

    # This 64 may need to change if the canvas size changes
    ymin = max(min(y-brush_width//2, 64 - brush_width), 0)
    ymax = ymin+brush_width

    # update output canvas
    output.coords(pixel_rect, 4*xmin, 4*ymin, 4*xmax, 4*ymax)
    output.tag_raise(pixel_rect)
    output.itemconfig(pixel_rect, outline=rgb(mycol[0], mycol[1], mycol[2]))

# Optional functions for the Neural Painter

# Localized Gaussian Smoothing Kernel
# Use this if you want changes to MASK to be more localized to the brush location in soe sense


def gk(c1, r1, c2, r2):
    # First, create X and Y arrays indicating distance to the boundaries of the paintbrush
    # In this current context, im is the ordinal number of pixels (64 typically)
    sigma = 0.3
    im = 64
    x = np.repeat([np.concatenate(
        [np.mgrid[-c1:0], np.zeros(c2-c1), np.mgrid[1:1+im-c2]])], im, axis=0)
    y = np.repeat(np.vstack(np.concatenate(
        [np.mgrid[-r1:0], np.zeros(r2-r1), np.mgrid[1:1+im-r2]])), im, axis=1)
    g = np.exp(-(x**2/float(im)+y**2/float(im))/(2*sigma**2))
    # remove the 3 if you want to apply this to mask rather than an RGB channel
    return np.repeat([g], 3, axis=0)
# This function reduces the likelihood of a change based on how close each individual pixel is to a maximal value.
# Consider conditioning this based on the gK value and the requested color. I.E. instead of just a flat distance from 128,
# have it be a difference from the expected color at a given location. This could also be used to "weight" the image towards staying the same.


def upperlim(image):
    h = 1
    return (1.0/((1.0/h)*np.abs(image-128)+1))

# Similar to upperlim, this function changes the value of the correction term if it's going to move pixels too close to a maximal value


def dampen(input, correct):
    # The closer input+correct is to -1 or 1, the further it is from 0.
    # We're okay with almost all values (i.e. between 0 and 0.8) but as 
    # we approach 1 we want to slow the change
    thresh = 0.75
    m = (input+correct) > thresh
    return -input*m+correct*(1-m)+thresh*m

# Neural Painter Function


def paint(event):
    global Z, output, myRGB, IM, ERROR, RECON, USER_MASK, SAMPLE_FLAG

    # Move the paintbrush
    move_mouse(event)
    print('Painting')

    # Define a gradient descent step-size
    weight = 0.05

    # Get paintbrush location
    [x1, y1, x2, y2] = [int(coordinate) //
                        4 for coordinate in output.coords(pixel_rect)]
    for _ in range(5):
        # Get dIM/dZ that minimizes the difference between IM and RGB in the domain of the paintbrush
        # print(myRGB.shape)
        # myRGB_temp = myRGB.swapaxes(1, 3).swapaxes(1, 2)
        # print(myRGB_temp.shape)
        temp = np.asarray(model.imgradRGB(x1, y1, x2, y2, np.float32(
            to_tanh(myRGB)), np.float32([Z.flatten()]))[0])
        grad = temp.reshape((10, 10))*(1+(x2-x1))

        # Update Z
        Z -= weight*grad

    # If operating on a sample, update sample
    if SAMPLE_FLAG:
        print('Updating with sample')
        update_canvas(w)
        update_photo(None, output)
    # Else, update photo
    else:
        print('Updating with photo')
        # Difference between current image and reconstruction
        DELTA = model.sample_at(np.float32([Z.flatten()]))[
            0]-to_tanh(np.float32(RECON))

        # Not-Yet-Implemented User Mask feature
        # USER_MASK[y1:y2,x1:x2]+=0.05

        # Get MASK
        MASK = scipy.ndimage.filters.gaussian_filter(
            np.min([np.mean(np.abs(DELTA), axis=0), np.ones((64, 64))], axis=0), 0.7)

        # Optionally dampen D
        # D = dampen(to_tanh(np.float32(RECON)),MASK*DELTA+(1-MASK)*ERROR)

        # Update image
        print('MASK')
        print(np.shape(MASK))
        print(np.min(MASK))
        print(np.max(MASK))
        print('ERROR')
        # ERROR = ERROR[[2, 0, 1], :, :]
        print(np.shape(ERROR))
        print(np.min(ERROR))
        print(np.max(ERROR))
        print('DELTA')
        print(np.shape(DELTA))
        print(np.min(DELTA))
        print(np.max(DELTA))
        print('RECON')
        print(np.shape(RECON))
        print(np.min(RECON))
        print(np.max(RECON))
        # DELTA[:] = 0
        D = MASK*DELTA+(1-MASK)*ERROR
        print('D')
        print(np.shape(D))
        print(np.min(D))
        print(np.max(D))

        # D = (1-MASK)*ERROR

        # D = D[[2,1,0],:,:]
        # D = D.swapaxes(0, 1).swapaxes(1, 2)
        print('D')
        print(D.shape)
        print(np.max(D))
        IM = from_tanh(to_tanh(RECON)+D)
        IM -= IM.min()
        IM /= IM.max()
        IM = np.uint8(from_tanh(IM))
        # IM = np.uint8(from_tanh(D))

        print('IMAGE')
        print(np.shape(IM))
        print(np.min(IM))
        print(np.max(IM))
        # Pass updates
        update_canvas(w)
        update_photo(IM, output)

# Load an image and infer/reconstruct from it. Update this with a function to load your own images if you want to edit
# non-celebA photos.


def infer():
    global Z, w, GIM, IM, ERROR, RECON, DELTA, USER_MASK, SAMPLE_FLAG, output
    val = myentry.get()
    try:
        val = int(val)
        GIM = np.asarray(np.load('CelebAValid2.npz')['arr_0'][val])
        IM = GIM
    except ValueError:
        print("No input")
        val = 420
        GIM = np.asarray(np.load('CelebAValid2.npz')['arr_0'][val])
        IM = GIM
    # myentry.delete(0, END) # Optionally, clear entry after typing it in

    # Reset Delta
    DELTA = np.zeros(np.shape(IM), dtype=np.float32)

    # Infer and reshape latents. This can be done without an intermediate variable if desired
    s = model.encode_images(np.asarray([to_tanh(IM)], dtype=np.float32))
    Z = np.reshape(s[0], np.shape(Z))

    # Get reconstruction
    RECON = np.uint8(from_tanh(model.sample_at(np.float32([Z.flatten()]))[0]))

    # Get error
    ERROR = to_tanh(np.float32(IM)) - to_tanh(np.float32(RECON))

    # Reset user mask
    USER_MASK *= 0

    # Clear the sample flag
    SAMPLE_FLAG = 0

    # Update photo
    update_photo(IM, output)
    update_canvas(w)

# Paint directly into the latent space


def paint_latents(event):
    global r, Z, output, painted_rects, MASK, USER_MASK, RECON

    # Get extent of latent paintbrush
    x1, y1 = (event.x - d.get()), (event.y - d.get())
    x2, y2 = (event.x + d.get()), (event.y + d.get())

    print('Painting Latents')
    selected_widget = event.widget

    # Paint in latent space and update Z
    painted_rects.append(event.widget.create_rectangle(
        x1, y1, x2, y2, fill=rb(color.get()), outline=rb(color.get())))
    r[max((y1-bd), 0):min((y2-bd), r.shape[0]), max((x1-bd), 0):min((x2-bd), r.shape[1])] = color.get()/255.0
    Z = np.asarray([np.mean(o) for v in [np.hsplit(h, Z.shape[0])
                                         for h in np.vsplit((r), Z.shape[1])]
                    for o in v]).reshape(Z.shape[0], Z.shape[1])
    if SAMPLE_FLAG:
        update_photo(None, output)
        # Remove this if you wish to see a more free-form paintbrush
        update_canvas(w)
    else:
        DELTA = model.sample_at(np.float32([Z.flatten()]))[
            0]-to_tanh(np.float32(RECON))
        MASK = scipy.ndimage.filters.gaussian_filter(
            np.min([np.mean(np.abs(DELTA), axis=0), np.ones((64, 64))], axis=0), 0.7)
        # D = dampen(to_tanh(np.float32(RECON)),MASK*DELTA+(1-MASK)*ERROR)
        D = MASK*DELTA+(1-MASK)*ERROR
        IM = np.uint8(from_tanh(to_tanh(RECON)+D))
        # Remove this if you wish to see a more free-form paintbrush
        update_canvas(w)
        update_photo(IM, output)

# Scroll to lighten or darken an image patch


def scroll(event):
    global r, Z, output
    # Optional alternate method to get a single X Y point
    # x,y = np.floor( ( event.x - (output.winfo_rootx() - master.winfo_rootx()) ) / 4), np.floor( ( event.y - (output.winfo_rooty() - master.winfo_rooty()) ) / 4)
    weight = 0.1
    [x1, y1, x2, y2] = [coordinate //
                        4 for coordinate in output.coords(pixel_rect)]
    grad = np.reshape(model.imgrad(x1, y1, x2, y2, np.float32([Z.flatten()]))[
                      0], Z.shape)*(1+(x2-x1))
    Z += np.sign(event.delta)*weight*grad
    update_canvas(w)
    update_photo(None, output)

# Samples in the latent space


def sample():
    global Z, output, RECON, IM, ERROR, SAMPLE_FLAG
    Z = np.random.randn(Z.shape[0], Z.shape[1]) # Get Gaussian Sample
    # Z = np.random.uniform(low=-1.0,high=1.0,size=(Z.shape[0],Z.shape[1])) # Optionally get uniform sample

    # Update reconstruction and error
    # RECON = (model.sample_at(np.float32([Z.flatten()]))[0])
    RECON = np.uint8(from_tanh(model.sample_at(np.float32([Z.flatten()]))[0]))
    ERROR = to_tanh(np.float32(IM)) - to_tanh(np.float32(RECON))
    update_canvas(w)
    SAMPLE_FLAG = 1
    update_photo(None, output)

# Reset to ground-truth image
def Reset():
    global GIM, IM, Z, DELTA, RECON, ERROR, USER_MASK, SAMPLE_FLAG, output
    IM = GIM
    Z = np.reshape(model.encode_images(np.asarray(
        [to_tanh(IM)], dtype=np.float32))[0], np.shape(Z))

    DELTA = np.zeros(np.shape(IM), dtype=np.float32)
    RECON = np.uint8(from_tanh(model.sample_at(np.float32([Z.flatten()]))[0]))
    print("shape of recon")

    print(RECON.shape)
    imsave('recon.png',np.float32(RECON.swapaxes(0,2).swapaxes(1,0))/255.0)
    ERROR = to_tanh(np.float32(IM)) - to_tanh(np.float32(RECON))
    USER_MASK *= 0
    SAMPLE_FLAG = 0
    update_canvas(w)
    update_photo(IM, output)


def UpdateGIM():
    global GIM, IM
    GIM = IM
    Reset()  # Recalc the latent space for the new ground-truth image.

# Change brush size


def update_brush(event):
    brush.create_rectangle(0, 0, 25, 25, fill=rgb(
        255, 255, 255), outline=rgb(255, 255, 255))
    brush.create_rectangle(int(12.5-d.get()/4.0), int(12.5-d.get()/4.0), int(
        12.5+d.get()/4.0), int(12.5+d.get()/4.0), fill=rb(color.get()), outline=rb(color.get()))

# assign color picker values to myRGB


def getColor():
    global myRGB, mycol  #, output

    col = askcolor((int(mycol[0]), int(mycol[1]), int(mycol[2])))
    if col[0] is None:
        return  # Dont change color if Cancel pressed.
    mycol = col[0]
    for i in range(0, 3):
        myRGB[0, i, :, :] = mycol[i]  # assign

# Optional function to "lock" latents so that gradients are always evaluated with respect to the locked Z
# def lock():
    # global Z,locked, Zlock, lockbutton
    # lockbutton.config(relief='raised' if locked else 'sunken')
    # Zlock = Z if not locked else Zlock
    # locked = not locked
# lockbutton = Button(f, text="Lock", command=lock,relief='raised')
# lockbutton.pack(side=LEFT)


# Prepare GUI
master.bind("<MouseWheel>", scroll)

# Prepare drawing canvas
f = Frame(master)
f.pack(side=TOP)
output = Canvas(f, name='output', width=64*4, height=64*4)
output.bind('<Motion>', move_mouse)
output.bind('<B1-Motion>', paint)
pixel_rect = output.create_rectangle(0, 0, 4, 4, outline='yellow')
output.pack()

# Prepare latent canvas
f = Frame(master, width=res*dim[0], height=dim[1]*10)
f.pack(side=TOP)
w = Canvas(f, name='canvas', width=res*dim[0], height=res*dim[1])
w.bind("<B1-Motion>", paint_latents)
# Produce painted rectangles
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        rects[i, j] = w.create_rectangle(
            j*res, i*res, (j+1)*res, (i+1)*res, fill=rb(255*Z[i, j]), outline=rb(255*Z[i, j]))
# w.create_rectangle( 0,0,res*dim[0],res*dim[1], fill = rgb(255,255,255),outline=rgb(255,255,255)) # Optionally Initialize canvas to white
w.pack()


# Color gradient
gradient = Canvas(master, width=400, height=20)
gradient.pack(side=TOP)
# gradient.grid(row=i+1)
for j in range(-200, 200):
    gradient.create_rectangle(
        j*255/200+200, 0, j*255/200+201, 20, fill=rb(j*255/200), outline=rb(j*255/200))
# Color scale slider
f = Frame(master)
Scale(master, from_=-255, to=255, length=canvas_width, variable=color,
      orient=HORIZONTAL, showvalue=0, command=update_brush).pack(side=TOP)

# Buttons and brushes
Button(f, text="Sample", command=sample).pack(side=LEFT)
Button(f, text="Reset", command=Reset).pack(side=LEFT)
Button(f, text="Update", command=UpdateGIM).pack(side=LEFT)
brush = Canvas(f, width=25, height=25)
Scale(f, from_=0, to=64, length=100, width=25, variable=d, orient=HORIZONTAL,
      showvalue=0, command=update_brush).pack(side=LEFT)  # Brush diameter scale
brush.pack(side=LEFT)
inferbutton = Button(f, text="Infer", command=infer)
inferbutton.pack(side=LEFT)
colorbutton = Button(f, text='Col', command=getColor)
colorbutton.pack(side=LEFT)
myentry = Entry()
myentry.pack(side=LEFT)
f.pack(side=TOP)


# Reset and infer to kick it off
Reset()
infer()
mainloop()
