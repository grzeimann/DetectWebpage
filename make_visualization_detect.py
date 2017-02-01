"""

@author: gregz

"""

from __future__ import print_function
import matplotlib
matplotlib.use('agg')
import argparse as ap
from pyhetdex.cure.distortion import Distortion
import pyhetdex.tools.files.file_tools as ft
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os.path as op
import os
import time
import textwrap
import CreateWebpage as CW
from collections import OrderedDict
from pyhetdex.het.ifu_centers import IFUCenter
from astropy.stats import biweight_location
from scipy.ndimage.filters import gaussian_filter
from astropy.modeling.models import Moffat2D, Gaussian2D
from photutils import CircularAperture, aperture_photometry
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy import units as u
from pyhetdex.het.fplane import FPlane
from pyhetdex.coordinates.tangent_projection_astropy import TangentPlane as TP
import glob
from datetime import datetime

plt.ioff()


dist_thresh = 2. # Fiber Distance
sn_cut = 4.0 # S/N Cut
xw = 24 # image width in x-dir 
yw = 10 # image width in y-dir
res = [3,9]
ww = xw*1.9 # wavelength width
contrast1 = 0.9 # convolved image
contrast2 = 0.5 # regular image
virus_config = '/work/03946/hetdex/maverick/virus_config'
fplane_file = '/home/00115/gebhardt/fplane.txt'
fplanedir = "/work/03946/hetdex/maverick/virus_config/fplane"
image_dir = '/work/03229/iwold/maverick/fall_field/stack/v2/psf/nano'
#virus_config = '/Users/gregz/cure/virus_early/virus_config'

SPECBIG = ["L","R"]

#
# CAM_IFUSLOT_DICT = {'004':'093',
#                     '037':'074',
#                     '027':'075',
#                     '047':'076',
#                     '024':'073',
#                     '013':'084',
#                     '016':'085',
#                     '041':'086',
#                     '051':'083',
#                     '008':'094',
#                     '025':'095',
#                     '038':'096',
#                     '020':'103',
#                     '032':'104',
#                     '012':'105',
#                     '017':'106',}
#
# IFUSLOT_DICT = {'073':['024','033'],
#                 '074':['037','024'],
#                 '075':['027','001'],
#                 '076':['047','016'],
#                 '083':['051','023'],
#                 '084':['013','019'],
#                 '085':['016','026'],
#                 '086':['041','015'],
#                 '093':['004','051'],
#                 '094':['008','054'],
#                 '095':['025','020'],
#                 '096':['038','014'],
#                 '103':['020','004'],
#                 '104':['032','028'],
#                 '105':['012','055'],
#                 '106':['017','022'],}
#
# # Dictionary of the mapping between SPECID and IFUID
#
# CAM_IFU_DICT = {'004':'051',
#                 '037':'024',
#                 '027':'001',
#                 '047':'016',
#                 '024':'033',
#                 '013':'019',
#                 '016':'026',
#                 '041':'015',
#                 '051':'023',
#                 '008':'054',
#                 '025':'020',
#                 '038':'014',
#                 '020':'004',
#                 '032':'028',
#                 '012':'055',
#                 '017':'022',}
                
# Default set of spectrographs for reduction
SPECID = ["004","008","012","013","016","017","020","024","025","027","032",
          "037","038","041","047","051"]
#SPECID = ["027"]
SIDE = ["L", "R"]

columnnames = ["SPECID", "NR", "ID", "S/N", "RA", "Dec", "Source_Info", "2D Plots","Spec Plots","Cutouts"]
columnnames_cont = ["SPECID", "ID", "S/N", "RA", "Dec", "2D Plots", "Spec Plots", "Cutouts"]


class EmisCatalog(object):
    @classmethod
    def writeHeader(cls, f):
        """Write the header to file ``f``

        Parameters
        ----------
        f : file-like object
            where to write to; must have a ``write`` method
        """
        s = []

        s.append("# "+str(datetime.now()))
        s.append("#col  1: ID ")
        s.append("#col  2: RA ")
        s.append("#col  3: DEC ")
        s.append("#col  4: wave ")
        s.append("#col  5: s/n ")
        s.append("#col  6: chi2 ")
        s.append("#col  7: flux ")

        f.write('\n'.join(s) + "\n")

    @classmethod
    def writeEmis(cls, f, source):
        if source is not None:
            s = ("%s %11.6f %11.6f %11.2f %11.2f %11.2f %11.2f" % (tuple(source)))
            f.write(s + "\n")
        f.flush()

def find_fplane(date): #date as yyyymmdd string
    """Locate the fplane file to use based on the observation date

        Parameters
        ----------
            date : string
                observation date as YYYYMMDD

        Returns
        -------
            fully qualified filename of fplane file
    """
    #todo: validate date

    filepath = fplanedir
    if filepath[-1] != "/":
        filepath += "/"
    files = glob.glob(filepath + "fplane*.txt")

    if len(files) > 0:
        target_file = filepath + "fplane" + date + ".txt"

        if target_file in files: #exact match for date, use this one
            fplane = target_file
        else:                   #find nearest earlier date
            files.append(target_file)
            files = sorted(files)
            #sanity check the index
            i = files.index(target_file)-1
            if i < 0: #there is no valid fplane
                print("Warning! No valid fplane file found for the given date. Will use oldest available.")
                i = 0
            fplane = files[i]
    else:
        print ("Error. No fplane files found.")

    return fplane

def build_fplane_dicts(fqfn):
    """Build the dictionaries maping IFUSLOTID, SPECID and IFUID

        Parameters
        ----------
        fqfn : string
            fully qualified file name of the fplane file to use

        Returns
        -------
            ifuslotid to specid, ifuid dictionary
            specid to ifuid dictionary
        """
    # IFUSLOT X_FP   Y_FP   SPECID SPECSLOT IFUID IFUROT PLATESC
    if fqfn is None:
        print("Error! Cannot build fplane dictionaries. No fplane file.")
        return {},{}

    ifuslot, specid, ifuid = np.loadtxt(fqfn, comments='#', usecols=(0, 3, 5), dtype = int, unpack=True)
    ifuslot_dict = {}
    cam_ifu_dict = {}
    cam_ifuslot_dict = {}

    for i in range(len(ifuslot)):
        if (ifuid[i] < 900) and (specid[i] < 900):
            ifuslot_dict[str("%03d" % ifuslot[i])] = [str("%03d" % specid[i]),str("%03d" % ifuid[i])]
            cam_ifu_dict[str("%03d" % specid[i])] = str("%03d" % ifuid[i])
            cam_ifuslot_dict[str("%03d" % specid[i])] = str("%03d" % ifuslot[i])

    return ifuslot_dict, cam_ifu_dict, cam_ifuslot_dict


class ParseDither():
    """
    Parse the dither file 

    Parameters
    ----------
    dither_file : string
        file containing the dither relative position.

    """

    def __init__(self, dither_file):
        self._absfname = op.abspath(dither_file)
        # common prefix of the L and R file names of the dither
        self.basename, self.deformer= [], []
        # delta x and y of the dithers
        self.dx, self.dy = [], []
        # image quality, illumination and airmass
        self.seeing, self.norm, self.airmass = [], [], []
        self._read_dither(dither_file)

    def _read_dither(self, dither_file):
        """
        Read the relative dither position

        Parameters
        ----------
        dither_file : string
            file containing the dither relative position.
        """
        with open(dither_file, 'r') as f:
            f = ft.skip_comments(f)
            for l in f:
                try:
                    _bn, _d, _x, _y, _seeing, _norm, _airmass = l.split()
                except ValueError:  # skip empty or incomplete lines
                    pass
                self.basename.append(_bn)
                self.deformer.append(_d)
                self.dx.append(float(_x))
                self.dy.append(float(_y))
                self.seeing.append(float(_seeing))
                self.norm.append(float(_norm))
                self.airmass.append(float(_airmass))

def parse_args(argv=None):
    """Parse the command line arguments

    Parameters
    ----------
    argv : list of string
        arguments to parse; if ``None``, ``sys.argv`` is used

    Returns
    -------
    Namespace
        parsed arguments
    """
    description = textwrap.dedent('''Visualizing Detect Catalog.''')
    parser = ap.ArgumentParser(description=description,
                            formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument("--folder", nargs='?', type=str, 
                        help='''Folder''', 
                        default=None)

    parser.add_argument("--dither_file", nargs='?', type=str, 
                        help='''Dither File''', 
                        default='dither.txt')

    parser.add_argument("-sd","--scidir_date", nargs='?', type=str,
                        help='''Science Directory Date. Ex: \"20160412\"''', default=None)

    parser.add_argument("--ra", nargs='?', type=float, 
                        help='''ra''', 
                        default=None)
                        
    parser.add_argument("--dec", nargs='?', type=float, 
                        help='''Dec''', 
                        default=None)

    parser.add_argument("--rot", nargs='?', type=float, 
                        help='''rotation''', 
                        default=None)
                        
    parser.add_argument("--webid", nargs='?', type=str, 
                        help='''WebID for joining later.''', 
                        default=None)
                        
    parser.add_argument("--specid", nargs='?', type=str, 
                        help='''List of SPECID's for processing. 
                        Ex: "020,008".''', default = None)

    parser.add_argument("--goodsn", help='''Goods-N?''',
                        action="count", default=0)   

    parser.add_argument("--cosmos", help='''COSMOS?''',
                        action="count", default=0)  

    parser.add_argument("--create_header", help='''Create Just Header.''',
                        action="count", default=0)  

    parser.add_argument("--create_ending", help='''Create Just Ending.''',
                        action="count", default=0) 

    parser.add_argument("--debug", help='''Debug''',
                        action="count", default=0)   
                                                                         
    args = parser.parse_args(args=argv) 

    # Check that the arguments are filled
    if not args.create_header and not args.create_ending:
        if args.ra is None:
            msg = 'No RA was provided'
            parser.error(msg)
        if args.dec is None:
            msg = 'No Dec was provided'
            parser.error(msg)
        if args.rot is None:
            msg = 'No Parangle was provided'
            parser.error(msg)
        if args.scidir_date is None:
            msg = 'No Science Directory Date was provided'
            parser.error(msg)
    if args.folder is None:
        msg = 'No folder was provided'
        parser.error(msg)
    # Check that the arguments are filled
    if args.specid:
        args.specid = args.specid.replace(" ", "").split(',')
    else:
        args.specid = SPECID 
    if args.create_header and args.create_ending:
        msg = 'Pick create_header or create_ending, not both'
        parser.error(msg)        
    return args
    
def pick_image(ra, dec):
    letters = ['A','B','C']
    numbers = np.arange(10)
    rai = []
    deci = []
    l = []
    n = []
    for let in letters:
        for number in numbers:
            num=str(int(number))
            fn = op.join(image_dir,'%s%s_g_sci.fits' %(let,num))
            if op.exists(fn):
                rai.append(fits.open(fn)[0].header['crval1'])
                deci.append(fits.open(fn)[0].header['crval2'])
                l.append(let)
                n.append(num)
    x = np.array(rai)
    y = np.array(deci)
    d = np.sqrt(((ra-x)*np.cos(dec*np.pi/180.))**2+(dec-y)**2)
    ind = np.argmin(d)
    filename = op.join(image_dir,'%s%s_g_sci.fits' %(l[ind],n[ind]))
    catname = op.join(op.dirname(image_dir),'%s%s_g_cat.fits' %(l[ind],n[ind]))
    return filename, catname
    
def get_w_as_r(seeing, gridsize, rstep, rmax, profile_name='moffat'):
    fradius = 0.75 # VIRUS
    if profile_name == 'moffat':
        alpha = 2.5 # hard coded in Cure
        gamma = seeing/2.0/np.sqrt(np.power(2.0,(1.0/alpha)) - 1.0)
        profile = Moffat2D(alpha = alpha, gamma = gamma)
    else:
        sigma = seeing/2.3548
        profile = Gaussian2D(x_stddev=sigma,y_stddev=sigma)             
    x = np.linspace(-1*(rmax+fradius+0.5),(rmax+fradius+0.5), gridsize)
    X,Y = np.meshgrid(x,x)
    Z = profile(X.ravel(),Y.ravel()).reshape(X.shape)
    Z /= np.sum(Z.ravel()*(x[1]-x[0])**2)
    nstep = int(rmax/rstep) + 1
    r = np.linspace(0, rmax, nstep)
    xloc = np.interp(r,x,np.arange(len(x)))
    yloc = np.interp(np.zeros((nstep,)),x,np.arange(len(x)))
    positions = [xloc,yloc]
    apertures = CircularAperture(positions, r=fradius)
    phot_table = aperture_photometry(Z, apertures)
    return r, np.array(phot_table['aperture_sum'])
    
def build_spec_image(datakeep, outfile, cwave, dwave=1.0, cmap=None, 
                     cmap2=None, debug=False):
    if not cmap:
        # Default cmap is gray
        cmap = plt.get_cmap('gray_r')
    if not cmap2:
        norm = plt.Normalize()
        colors = plt.cm.hsv(norm(np.arange(len(datakeep['ra'])+2)))
    N = len(datakeep['xi'])
    rm = 0.2
    fig = plt.figure(figsize=(5,3))
    r, w = get_w_as_r(1.5,500,0.05,6.)
    specplot = plt.axes([0.1, 0.1, 0.8, 0.8])
    bigwave = np.arange(cwave-ww,cwave+ww+dwave,dwave)
    F = np.zeros(bigwave.shape)
    mn = 100.0
    mx = 0.0
    W = 0.0
    ind = sorted(range(len(datakeep['d'])), key=lambda k: datakeep['d'][k], 
                 reverse=True)
    for i in xrange(N):

        specplot.step(datakeep['specwave'][ind[i]], datakeep['spec'][ind[i]], 
                          where='mid', color=colors[i,0:3],alpha=0.5)
        w1 = np.interp(datakeep['d'][ind[i]],r,w)
        F+=(np.interp(bigwave,datakeep['specwave'][ind[i]], datakeep['spec'][ind[i]])*w1) #*(2.0 -datakeep['d'][ind[i]])
        W+=w1
        mn = np.min([mn,np.min(datakeep['spec'][ind[i]])])
        mx = np.max([mx,np.max(datakeep['spec'][ind[i]])])
    F /= W
    specplot.step(bigwave, F, c='b',where='mid',lw=2)
    ran = mx - mn
    specplot.errorbar(cwave-.8*ww, mn+ran*(1+rm)*0.85, 
                      yerr=biweight_location(np.array(datakeep['spece'][:])),
                      fmt='o',marker='o', ms=4, mec='k', ecolor=[0.7,0.7,0.7],
                      mew=1, elinewidth=3, mfc=[0.7,0.7,0.7])
    specplot.plot([cwave,cwave],[mn-ran*rm, mn+ran*(1+rm)],ls='--',c=[0.3,0.3,0.3])
    specplot.axis([cwave-ww, cwave+ww, mn-ran*rm, mn+ran*(1+rm)])
    fig.savefig(outfile,dpi=150)
    plt.close(fig)


def make_image_cutout(datakeep, data, wcs, ras, decs, outfile, cmap2=None,
                      cmap=None, sz=15., debug=False, args=None, cat=None,
                      within=None):
    if not cmap:
        # Default cmap is gray
        cmap = plt.get_cmap('gray_r')
    if not cmap2:
        norm = plt.Normalize()
        colors = plt.cm.hsv(norm(np.arange(len(datakeep['ra'])+2)))
    pixsize_x = np.sqrt(wcs.wcs.cd[0,0]**2 + wcs.wcs.cd[0,1]**2)*3600. 
    ind = sorted(range(len(datakeep['d'])), key=lambda k: datakeep['d'][k], 
                 reverse=True)
    size = int(sz / pixsize_x)
    position = SkyCoord(ras, decs, unit="deg", frame='fk5')
    cutout = Cutout2D(data, position, (size,size), wcs=wcs)
    fig = plt.figure(figsize=(4,4))
    if args.goodsn:
        vmin = -0.02
        vmax = 0.08
    elif args.cosmos:
        vmin = -11
        vmax = 12
    else:
        vmin = -10
        vmax = 50
    plt.imshow(cutout.data,origin='lower',interpolation='nearest',vmin=vmin,vmax=vmax, 
               cmap=cmap, extent=[-sz/2.,sz/2.,-sz/2.,sz/2.])
    xc, yc = skycoord_to_pixel(position, wcs=cutout.wcs)
    if cat is not None:
        for i in xrange(len(within)):
            position2 = SkyCoord(cat['alpha_j2000'][within[i]], 
                                 cat['delta_j2000'][within[i]], unit="deg", 
                                 frame='fk5')   
            xc2, yc2 = skycoord_to_pixel(position2, wcs=cutout.wcs)
            plt.scatter((xc2-xc)*pixsize_x, (yc2-yc)*pixsize_x, marker='x', c='g', s=35)
    plt.scatter(0., 0.,marker='x',c='r',s=35)
    circle = plt.Circle((0., 0.), radius=2., fc='none', 
                            ec='r', zorder=2, alpha=1.0)
    plt.gca().add_patch(circle)
    num = len(datakeep['ra'])
    for i in xrange(num):
        xf,yf = skycoord_to_pixel(
             SkyCoord(datakeep['ra'][ind[i]],datakeep['dec'][ind[i]], unit="deg", frame='fk5'), 
             wcs=cutout.wcs)
        circle = plt.Circle(((xf-xc)*pixsize_x, (yf-yc)*pixsize_x), radius=.75, fc='none', 
                            ec=colors[i,0:3], zorder=2, alpha=1.0)
        plt.text((xf-xc)*pixsize_x, (yf-yc)*pixsize_x, num-i, ha='center', va='center',
                 fontsize='x-small', color=colors[i,0:3]) #datakeep['fib'][ind[i]]
        plt.gca().add_patch(circle)
    fig.savefig(outfile,dpi=150)
    plt.close(fig)        
    
def build_2d_image(datakeep, outfile, cmap=None, cmap2=None, debug=False):
    if not cmap:
        # Default cmap is gray
        cmap = plt.get_cmap('gray_r')
    if not cmap2:
        norm = plt.Normalize()
#        colors = plt.cm.viridis_r(norm(np.arange(len(datakeep['ra'])+2)))
        colors = plt.cm.hsv(norm(np.arange(len(datakeep['ra']) + 2)))
    num = len(datakeep['xi'])
    bordbuff = 0.01
    borderxl = 0.05
    borderxr = 0.15
    borderyb = 0.05
    borderyt = 0.15
    dx = (1. - borderxl - borderxr) / 3.
    dy = (1. - borderyb - borderyt) / num
    dx1 = (1. - borderxl - borderxr) / 3.
    dy1 = (1. - borderyb - borderyt-num*bordbuff) / num
    Y = (yw / dy) / (xw / dx) * 5.

    fig = plt.figure(figsize=(5,Y),frameon=False)

    ind = sorted(range(len(datakeep['d'])), key=lambda k: datakeep['d'][k], 
                 reverse=True)
    for i in xrange(num):
        borplot = plt.axes([borderxl+0.*dx, borderyb+i*dy, 3*dx, dy])
        implot = plt.axes([borderxl+2.*dx-bordbuff/3., borderyb+i*dy+bordbuff/2., dx1, dy1])
        errplot = plt.axes([borderxl+1.*dx+1*bordbuff/3., borderyb+i*dy+bordbuff/2., dx1, dy1])
        cosplot = plt.axes([borderxl+0.*dx+bordbuff/2., borderyb+i*dy+bordbuff/2., dx1, dy1])
        autoAxis = borplot.axis()
        rec = plt.Rectangle((autoAxis[0]+bordbuff/2.,autoAxis[2]+bordbuff/2.),(autoAxis[1]-autoAxis[0])*(1.-bordbuff),
                            (autoAxis[3]-autoAxis[2])*(1.-bordbuff), fill=False, lw=3, 
                            color = colors[i,0:3], zorder=1)
        rec = borplot.add_patch(rec)
        borplot.set_xticks([])
        borplot.set_yticks([]) 
        borplot.axis('off')
        ext = list(np.hstack([datakeep['xl'][ind[i]],datakeep['xh'][ind[i]],
                              datakeep['yl'][ind[i]],datakeep['yh'][ind[i]]]))
        GF = gaussian_filter(datakeep['im'][ind[i]],(2,1))  
        implot.imshow(GF, 
                      origin="lower", cmap=cmap, 
                      interpolation="nearest",vmin=datakeep['vmin1'][ind[i]],
                      vmax=datakeep['vmax1'][ind[i]],
                      extent=ext)
        implot.scatter(datakeep['xi'][ind[i]],datakeep['yi'][ind[i]],
                       marker='.',c='r', edgecolor='r',s=10)
        implot.set_xticks([])
        implot.set_yticks([])
        implot.axis(ext)
        implot.axis('off')
        errplot.imshow(datakeep['pix'][ind[i]], 
                      origin="lower", cmap=plt.get_cmap('gray'), 
                      interpolation="nearest",vmin=0.9,vmax=1.1,
                      extent=ext)
        errplot.scatter(datakeep['xi'][ind[i]],datakeep['yi'][ind[i]],
                       marker='.',c='r', edgecolor='r', s=10)
        errplot.set_xticks([])
        errplot.set_yticks([])
        errplot.axis(ext)
        errplot.axis('off')
        a = datakeep['cos'][ind[i]]
        a = np.ma.masked_where(a==0, a)
        cmap1 = cmap
        cmap1.set_bad(color=[0.2,1.0,0.23])
        cosplot.imshow(a, 
                      origin="lower",cmap=cmap1,
                      interpolation="nearest",vmin=datakeep['vmin2'][ind[i]],
                      vmax=datakeep['vmax2'][ind[i]],
                      extent=ext)
        cosplot.scatter(datakeep['xi'][ind[i]],datakeep['yi'][ind[i]],
                       marker='.',c='r', edgecolor='r', s=10)
        cosplot.set_xticks([])
        cosplot.set_yticks([])
        cosplot.axis(ext)
        cosplot.axis('off')
        xi = datakeep['xi'][ind[i]]
        yi = datakeep['yi'][ind[i]]
        xl = int(np.round(xi-ext[0]-res[0]/2.))
        xh = int(np.round(xi-ext[0]+res[0]/2.))
        yl = int(np.round(yi-ext[2]-res[0]/2.))
        yh = int(np.round(yi-ext[2]+res[0]/2.))
        S = np.where(datakeep['err'][ind[i]][yl:yh,xl:xh]<0,0.,datakeep['im'][ind[i]][yl:yh,xl:xh]).sum()
        N = np.sqrt(np.where(datakeep['err'][ind[i]][yl:yh,xl:xh]<0,0.,datakeep['err'][ind[i]][yl:yh,xl:xh]**2).sum())
        sn = S/N

        implot.text(0.9, .75, num-i,
                    transform=implot.transAxes, fontsize=6, color=colors[i,0:3],
                    verticalalignment='bottom', horizontalalignment='left')

        implot.text(1.10,.75,'S/N = %0.2f' %(sn),
                    transform=implot.transAxes,fontsize=6,color='r',
                    verticalalignment='bottom', horizontalalignment='left')
        implot.text(1.10,.55,'D(") = %0.2f' %(datakeep['d'][ind[i]]),
                    transform=implot.transAxes,fontsize=6,color='r',
                    verticalalignment='bottom', horizontalalignment='left')
        implot.text(1.10, .35, 'X,Y = %d,%d' % (datakeep['xi'][ind[i]],datakeep['yi'][ind[i]]),
                    transform=implot.transAxes, fontsize=6, color='b',
                    verticalalignment='bottom', horizontalalignment='left')
        implot.text(1.10, .15, 'D,S,F = %d,%s,%d' % (datakeep['dit'][ind[i]],datakeep['side'][ind[i]],
                                                     datakeep['fib'][ind[i]]),
                    transform=implot.transAxes, fontsize=6, color='b',
                    verticalalignment='bottom', horizontalalignment='left')
        if i==(N-1):
            implot.text(0.5,.85,'Image',
                    transform=implot.transAxes,fontsize=8,color='b',
                    verticalalignment='bottom', horizontalalignment='center')
            errplot.text(0.5,.85,'Error',
                    transform=errplot.transAxes,fontsize=8,color='b',
                    verticalalignment='bottom', horizontalalignment='center')
            cosplot.text(0.5,.85,'Mask',
                    transform=cosplot.transAxes,fontsize=8,color='b',
                    verticalalignment='bottom', horizontalalignment='center') 
      
    fig.savefig(outfile,dpi=150)
    plt.close(fig)

def decimal_to_hours(ra, dec):
    rah = int(ra / 15.)
    ram = int((ra / 15. - rah) * 60.)
    ras = ((ra / 15. - rah) * 60. - ram) * 60.
    RA = '%02d:%02d:%02.2f' %(rah, ram, ras)
    if dec>0:
        decd = int(dec)
        decm = int((dec - decd) * 60.)
        decs = ((dec - decd) * 60. - decm) * 60.
        DEC = '%02d:%02d:%05.2f' %(decd, decm, decs)
    else:
        decd = int(dec)
        decm = int((decd - dec) * 60.)
        decs = (((decd - dec) * 60.) - decm) * 60. 
        DEC = '-%02d:%02d:%05.2f' %(np.abs(decd), decm, decs)
    return RA, DEC
    


def make_emission_row(Cat, f_webpage, args, D, Di, ifux, ifuy, IFU, tp, specid, 
                      wcs, data):
    for i, a  in enumerate(Cat['XS']):
        if args.debug:
            t1 = time.time()
        x = Cat['XS'][i]
        y = Cat['YS'][i]
        sn = Cat['sigma'][i]
        chi2 = Cat['chi2'][i]
        flux = Cat['dataflux'][i]
        eqw = Cat['eqw'][i]
        datakeep = {}
        datakeep['dit'] = []
        datakeep['side'] = []
        datakeep['fib'] = []
        datakeep['xi'] = []
        datakeep['yi'] = []
        datakeep['xl'] = []
        datakeep['yl'] = []
        datakeep['xh'] = []
        datakeep['yh'] = []
        datakeep['sn'] = []
        datakeep['d'] = []
        datakeep['dx'] = []
        datakeep['dy'] = []
        datakeep['im'] = []
        datakeep['vmin1'] = []
        datakeep['vmax1'] = []
        datakeep['vmin2'] = []
        datakeep['vmax2'] = []
        datakeep['err'] = []
        datakeep['pix'] = []
        datakeep['spec'] = []
        datakeep['spece'] = []
        datakeep['specwave'] = []
        datakeep['cos'] = []
        datakeep['par'] = []
        datakeep['ra'] = []
        datakeep['dec'] = []
        ras, decs = tp.xy2raDec(x+ifuy,y+ifux)
        if sn>sn_cut:
            for side in SIDE:
                for dither in xrange(len(Di.dx)):
                    dx = x-IFU.xifu[side]+Di.dx[dither]
                    dy = y-IFU.yifu[side]+Di.dy[dither]
                    d = np.sqrt(dx**2 + dy**2)
                    loc = np.where(d<dist_thresh)[0]
                    for l in loc:
                        datakeep['dit'].append(dither +1)
                        datakeep['side'].append(side)
                        f0 = D[side].get_reference_f(l+1)
                        xi = D[side].map_wf_x(Cat['l'][i],f0)
                        yi = D[side].map_wf_y(Cat['l'][i],f0)
                        datakeep['fib'].append(D[side].map_xy_fibernum(xi, yi))
                        xfiber = IFU.xifu[side][l]-Di.dx[dither]
                        yfiber = IFU.yifu[side][l]-Di.dy[dither]
                        xfiber += ifuy
                        yfiber += ifux
                        ra, dec = tp.xy2raDec(xfiber, yfiber)
                        datakeep['ra'].append(ra)
                        datakeep['dec'].append(dec)
                        xl = int(np.round(xi-xw))
                        xh = int(np.round(xi+xw))
                        yl = int(np.round(yi-yw))
                        yh = int(np.round(yi+yw))
                        datakeep['xi'].append(xi)
                        datakeep['yi'].append(yi)
                        datakeep['xl'].append(xl)
                        datakeep['yl'].append(yl)
                        datakeep['xh'].append(xh)
                        datakeep['yh'].append(yh)
                        datakeep['d'].append(d[l])
                        datakeep['sn'].append(sn)
                        dir_fn = op.dirname(Di.basename[dither])
                        base_fn = op.basename(Di.basename[dither])
                        if args.debug:
                            print(xi,yi,base_fn+'_'+side+'.fits')
                        im_fn = op.join(args.folder, 'c'+specid, op.join(
                                         dir_fn, base_fn+'_'+side+'.fits'))
                        err_fn = op.join(args.folder, 'c'+specid, op.join(
                                         dir_fn, 'e.'+base_fn+'_'+side+'.fits'))
                        cos_fn = op.join(args.folder, 'c'+specid, op.join(
                                         dir_fn, 'c'+base_fn+'_'+side+'.fits'))
                        FE_fn = op.join(args.folder, 'c'+specid, op.join(
                                         dir_fn, 'Fe'+base_fn+'_'+side+'.fits'))
                        FEe_fn = op.join(args.folder, 'c'+specid, op.join(
                                         dir_fn, 'e.Fe'+base_fn+'_'+side+'.fits'))
                        pix_fn = op.join(virus_config, 'PixelFlats','20161223',
                                         'pixelflat_cam%s_%s.fits'%(specid,side)) 
                        if op.exists(im_fn):
                            datakeep['im'].append(fits.open(im_fn)[0].data[yl:yh,xl:xh])
                            datakeep['par'].append(fits.open(im_fn)[0].header['PARANGLE'])
                            I = fits.open(im_fn)[0].data.ravel()
                            I[np.isnan(I)] = 0.0
                            s_ind = np.argsort(I)
                            len_s = len(s_ind)
                            s_rank = np.arange(len_s)
                            p = np.polyfit(s_rank-len_s/2,I[s_ind],1)
                            z1 = I[s_ind[len_s/2]]+p[0]*(1-len_s/2)/contrast1
                            z2 = I[s_ind[len_s/2]]+p[0]*(len_s-len_s/2)/contrast1
                            datakeep['vmin1'].append(z1)
                            datakeep['vmax1'].append(z2)
                            z1 = I[s_ind[len_s/2]]+p[0]*(1-len_s/2)/contrast2
                            z2 = I[s_ind[len_s/2]]+p[0]*(len_s-len_s/2)/contrast2
                            datakeep['vmin2'].append(z1)
                            datakeep['vmax2'].append(z2)                            
                        if op.exists(err_fn):
                            datakeep['err'].append(fits.open(err_fn)[0].data[yl:yh,xl:xh])
                        if op.exists(pix_fn):
                            datakeep['pix'].append(fits.open(pix_fn)[0].data[yl:yh,xl:xh])
                        if op.exists(cos_fn):
                            datakeep['cos'].append(fits.open(cos_fn)[0].data[yl:yh,xl:xh])
                        if op.exists(FE_fn):
                            FE = fits.open(FE_fn)[0].data
                            FEe = fits.open(FEe_fn)[0].data
                            nfib, xlen = FE.shape
                            crval = fits.open(FE_fn)[0].header['CRVAL1']
                            cdelt = fits.open(FE_fn)[0].header['CDELT1']
                            wave = np.arange(xlen)*cdelt + crval
                            Fe_indl = np.searchsorted(wave,Cat['l'][i]-ww,side='left')
                            Fe_indh = np.searchsorted(wave,Cat['l'][i]+ww,side='right')
                            datakeep['spec'].append(FE[l,Fe_indl:(Fe_indh+1)])
                            datakeep['spece'].append(FEe[l,Fe_indl:(Fe_indh+1)])
                            datakeep['specwave'].append(wave[Fe_indl:(Fe_indh+1)])
            if datakeep['xi']:
                if args.debug:
                    t2 = time.time()
                    print("Time Taken Building Source Dictionary: %0.2f" %(t2-t1))
                    t1 = time.time()
                outfile_2d = ('images/image2d_%s_specid_%s_object_%i_%i.png' 
                            % (op.basename(args.folder), specid, Cat['NR'][i], 
                               Cat['ID'][i]))
                build_2d_image(datakeep, outfile_2d, debug=args.debug)
                if args.debug:
                    t2 = time.time()
                    print("Time Taken Making 2d Image: %0.2f" %(t2-t1))
                    t1 = time.time()
                outfile_spec = ('images/imagespec_%s_specid_%s_object_%i_%i.png' 
                            % (op.basename(args.folder), specid, Cat['NR'][i],
                               Cat['ID'][i]))
                build_spec_image(datakeep, outfile_spec, 
                                 cwave=Cat['l'][i], debug=args.debug)  
                if args.debug:
                    t2 = time.time()
                    print("Time Taken Making Spectra Plot: %0.2f" %(t2-t1))
                    t1 = time.time()
                outfile_cut = ('images/imagecut_%s_specid_%s_object_%i_%i.png' 
                            % (op.basename(args.folder), specid, Cat['NR'][i], 
                               Cat['ID'][i]))
                make_image_cutout(datakeep, data, wcs, ras, decs, 
                                  outfile_cut, debug=args.debug, args=args)
                if args.debug:
                    t2 = time.time()
                    print("Time Taken Making Image Cutout: %0.2f" %(t2-t1))
                dict_web = OrderedDict()
                dict_web['Number_1'] = int(specid)
                dict_web['Number_2'] = int(Cat['NR'][i])
                dict_web['Number_3'] = int(Cat['ID'][i])
                dict_web['Number_4'] = sn
                dict_web['Number_5'] = float(ras)
                dict_web['Number_6'] = float(decs)
                RA, DEC = decimal_to_hours(ras, decs)
                dict_web['Table_1'] = [('S/N: %0.2f' %(sn)),
                                       ('chi2: %0.2f' %(chi2)),
                                       ('flux: %0.1f'% (flux)),
                                       ('eqw: %0.1f' %(eqw)),
                                       ('RA: %s' %RA), 
                                       ('Dec: %s' %DEC),
                                       ('X: %0.2f, Y: %0.2f' %(x, y))]
                dict_web['Image_1'] = outfile_2d
                dict_web['Image_2'] = outfile_spec
                dict_web['Image_3'] = outfile_cut
                CW.CreateWebpage.writeColumn(f_webpage,dict_web)
                EmisCatalog.writeEmis(f_emis, ['%s_%s'%(specid, Cat['ID'][i]),
                                               ras, decs, Cat['l'][i], sn, 
                                               chi2, flux])



def make_continuum_row(Cat, f_webpage, args, D, Di, ifux, ifuy, IFU, tp, specid, 
                      wcs, data, catalog, field, match_catalog):
    with open(match_catalog,'a') as f_match:
        for i, a  in enumerate(Cat['icx']):
            if args.debug:
                t1 = time.time()
            x = Cat['icx'][i]
            y = Cat['icy'][i]
            sn = Cat['sigma'][i]
            datakeep = {}
            datakeep['dit'] = []
            datakeep['side'] = []
            datakeep['fib'] = []
            datakeep['xi'] = []
            datakeep['yi'] = []
            datakeep['xl'] = []
            datakeep['yl'] = []
            datakeep['xh'] = []
            datakeep['yh'] = []
            datakeep['sn'] = []
            datakeep['d'] = []
            datakeep['dx'] = []
            datakeep['dy'] = []
            datakeep['im'] = []
            datakeep['vmin1'] = []
            datakeep['vmax1'] = []
            datakeep['vmin2'] = []
            datakeep['vmax2'] = []
            datakeep['err'] = []
            datakeep['pix'] = []
            datakeep['spec'] = []
            datakeep['spece'] = []
            datakeep['specwave'] = []
            datakeep['cos'] = []
            datakeep['par'] = []
            datakeep['ra'] = []
            datakeep['dec'] = []
            ras, decs = tp.xy2raDec(x+ifuy,y+ifux)
            c = SkyCoord(ras, decs, unit=(u.degree, u.degree))
            if args.debug:
                t1 = time.time()
            cat = SkyCoord(catalog['alpha_j2000'],catalog['delta_j2000'], 
                       unit=(u.degree, u.degree))
            idx, d2d, d3d = match_coordinates_sky(c, cat, nthneighbor=1)
            idx2, d2d2, d3d2 = match_coordinates_sky(c, cat, nthneighbor=2)
            
            within = []
            if d2d.arcsec[0] < 5.:
                within.append(idx)
                f_match.write('%s   %s   %s   %s   %s   %s   %s   %s\n'
                        % ((field).ljust(9), (specid).ljust(9), ("%0.5f" %ras).ljust(9), 
                           ("%0.5f" %decs).ljust(8), 
                           ("%0.5f" %catalog['alpha_j2000'][idx]).ljust(9),
                           ("%0.5f" %catalog['delta_j2000'][idx]).ljust(8), 
                           ("%0.3f" %(catalog['mag_auto'][idx]+25.07)).ljust(8),
                           ("%0.2f" %d2d.arcsec[0]).ljust(4)))
                f_match.flush()
               
            if d2d2.arcsec[0] < 5.:
                within.append(idx2)  
                f_match.write('%s   %s   %s   %s   %s   %s   %s   %s\n'
                        % ((field).ljust(9), (specid).ljust(9), ("%0.5f" %ras).ljust(9), 
                           ("%0.5f" %decs).ljust(8), 
                           ("%0.5f" %catalog['alpha_j2000'][idx2]).ljust(9),
                           ("%0.5f" %catalog['delta_j2000'][idx2]).ljust(8), 
                           ("%0.3f" %(catalog['mag_auto'][idx2]+25.07)).ljust(8),
                           ("%0.2f" %d2d2.arcsec[0]).ljust(4)))
                f_match.flush()
            if args.debug:
                t2 = time.time()
                print("Time Taken matching catalogs: %0.2f" %(t2-t1))
                print(d2d.arcsec[0], d2d2.arcsec[0])
                print(catalog['alpha_j2000'][idx], catalog['delta_j2000'][idx], 
                      catalog['mag_auto'][idx])
            if sn>1:
                for side in SIDE:
                    for dither in xrange(len(Di.dx)):
                        dx = x-IFU.xifu[side]+Di.dx[dither]
                        dy = y-IFU.yifu[side]+Di.dy[dither]
                        d = np.sqrt(dx**2 + dy**2)
                        loc = np.where(d<dist_thresh)[0]
                        for l in loc:
                            datakeep['dit'].append(dither + 1)
                            datakeep['side'].append(side)
                            f0 = D[side].get_reference_f(l+1)
                            xi = D[side].map_wf_x(Cat['zmin'][i]/2.+Cat['zmax'][i]/2.,f0)
                            yi = D[side].map_wf_y(Cat['zmin'][i]/2.+Cat['zmax'][i]/2.,f0)
                            datakeep['fib'].append(D[side].map_xy_fibernum(xi, yi))
                            xfiber = IFU.xifu[side][l]-Di.dx[dither]
                            yfiber = IFU.yifu[side][l]-Di.dy[dither]
                            xfiber += ifuy
                            yfiber += ifux
                            ra, dec = tp.xy2raDec(xfiber, yfiber)
                            datakeep['ra'].append(ra)
                            datakeep['dec'].append(dec)
                            xl = int(np.round(xi-xw))
                            xh = int(np.round(xi+xw))
                            yl = int(np.round(yi-yw))
                            yh = int(np.round(yi+yw))
                            datakeep['xi'].append(xi)
                            datakeep['yi'].append(yi)
                            datakeep['xl'].append(xl)
                            datakeep['yl'].append(yl)
                            datakeep['xh'].append(xh)
                            datakeep['yh'].append(yh)
                            datakeep['d'].append(d[l])
                            datakeep['sn'].append(sn)
                            dir_fn = op.dirname(Di.basename[dither])
                            base_fn = op.basename(Di.basename[dither])
                            if args.debug:
                                print(xi[0],yi[0],base_fn+'_'+side+'.fits')
                            im_fn = op.join(args.folder, 'c'+specid, op.join(
                                             dir_fn, base_fn+'_'+side+'.fits'))
                            err_fn = op.join(args.folder, 'c'+specid, op.join(
                                             dir_fn, 'e.'+base_fn+'_'+side+'.fits'))
                            cos_fn = op.join(args.folder, 'c'+specid, op.join(
                                             dir_fn, 'c'+base_fn+'_'+side+'.fits'))
                            FE_fn = op.join(args.folder, 'c'+specid, op.join(
                                             dir_fn, 'Fe'+base_fn+'_'+side+'.fits'))
                            FEe_fn = op.join(args.folder, 'c'+specid, op.join(
                                             dir_fn, 'e.Fe'+base_fn+'_'+side+'.fits'))
                            pix_fn = op.join(virus_config, 'PixelFlats','20161223',
                                             'pixelflat_cam%s_%s.fits'%(specid,side)) 
                            if op.exists(im_fn):
                                datakeep['im'].append(fits.open(im_fn)[0].data[yl:yh,xl:xh])
                                datakeep['par'].append(fits.open(im_fn)[0].header['PARANGLE'])
                                I = fits.open(im_fn)[0].data.ravel()
                                I[np.isnan(I)] = 0.0
                                s_ind = np.argsort(I)
                                len_s = len(s_ind)
                                s_rank = np.arange(len_s)
                                p = np.polyfit(s_rank-len_s/2,I[s_ind],1)
                                z1 = I[s_ind[len_s/2]]+p[0]*(1-len_s/2)/contrast1
                                z2 = I[s_ind[len_s/2]]+p[0]*(len_s-len_s/2)/contrast1
                                datakeep['vmin1'].append(z1)
                                datakeep['vmax1'].append(z2)
                                z1 = I[s_ind[len_s/2]]+p[0]*(1-len_s/2)/contrast2
                                z2 = I[s_ind[len_s/2]]+p[0]*(len_s-len_s/2)/contrast2
                                datakeep['vmin2'].append(z1)
                                datakeep['vmax2'].append(z2)
                            if op.exists(err_fn):
                                datakeep['err'].append(fits.open(err_fn)[0].data[yl:yh,xl:xh])
                            if op.exists(pix_fn):
                                datakeep['pix'].append(fits.open(pix_fn)[0].data[yl:yh,xl:xh])
                            if op.exists(cos_fn):
                                datakeep['cos'].append(fits.open(cos_fn)[0].data[yl:yh,xl:xh])
                            if op.exists(FE_fn):
                                FE = fits.open(FE_fn)[0].data
                                FEe = fits.open(FEe_fn)[0].data
                                nfib, xlen = FE.shape
                                crval = fits.open(FE_fn)[0].header['CRVAL1']
                                cdelt = fits.open(FE_fn)[0].header['CDELT1']
                                wave = np.arange(xlen)*cdelt + crval
                                Fe_indl = np.searchsorted(wave,Cat['zmin'][i]/2.+Cat['zmax'][i]/2.-ww,side='left')
                                Fe_indh = np.searchsorted(wave,Cat['zmin'][i]/2.+Cat['zmax'][i]/2.+ww,side='right')
                                datakeep['spec'].append(FE[l,Fe_indl:(Fe_indh+1)])
                                datakeep['spece'].append(FEe[l,Fe_indl:(Fe_indh+1)])
                                datakeep['specwave'].append(wave[Fe_indl:(Fe_indh+1)])
                if datakeep['xi']:
                    if args.debug:
                        t2 = time.time()
                        print("Time Taken Building Source Dictionary: %0.2f" %(t2-t1))
                        t1 = time.time()
                    outfile_2d = ('images/image2d_cont_%s_specid_%s_object_%i.png' 
                                % (op.basename(args.folder), specid, 
                                   Cat['ID'][i]))
                    build_2d_image(datakeep, outfile_2d, debug=args.debug)
                    if args.debug:
                        t2 = time.time()
                        print("Time Taken Making 2d Image: %0.2f" %(t2-t1))
                        t1 = time.time()
                    outfile_spec = ('images/imagespec_cont_%s_specid_%s_object_%i.png' 
                                % (op.basename(args.folder), specid, 
                                   Cat['ID'][i]))
                    build_spec_image(datakeep, outfile_spec, 
                                     cwave=Cat['zmin'][i]/2.+Cat['zmax'][i]/2., debug=args.debug)  
                    if args.debug:
                        t2 = time.time()
                        print("Time Taken Making Spectra Plot: %0.2f" %(t2-t1))
                        t1 = time.time()
                    outfile_cut = ('images/imagecut_cont_%s_specid_%s_object_%i.png' 
                                % (op.basename(args.folder), specid, 
                                   Cat['ID'][i]))
                    make_image_cutout(datakeep, data, wcs, ras, decs, 
                                      outfile_cut, debug=args.debug, args=args,
                                      cat=catalog, 
                                      within=within)
                    if args.debug:
                        t2 = time.time()
                        print("Time Taken Making Image Cutout: %0.2f" %(t2-t1))
                    dict_web = OrderedDict()
                    dict_web['Number_1'] = int(specid)
                    dict_web['Number_2'] = int(Cat['ID'][i])
                    dict_web['Number_3'] = sn
                    dict_web['Number_4'] = float(ras)
                    dict_web['Number_5'] = float(decs)
                    dict_web['Image_1'] = outfile_2d
                    dict_web['Image_2'] = outfile_spec
                    dict_web['Image_3'] = outfile_cut
                    CW.CreateWebpage.writeColumn(f_webpage, dict_web)  
   
def main():
    args = parse_args()

    #todo: get the date (YYYYMMDD) ... added argument, but is there a better way?
    if args.scidir_date is not None:
        fplane_file = find_fplane(args.scidir_date)
        IFUSLOT_DICT, CAM_IFU_DICT, CAM_IFUSLOT_DICT = build_fplane_dicts(fplane_file)

    if args.webid is None:
        webpage_name = 'Detect_Visualization_' + op.basename(args.folder)+'_emis'
        match_catalog = 'continuum_matches_'+ op.basename(args.folder) + '.dat'
        emis_catalog = 'emission_catalog_'+ op.basename(args.folder) + '.dat'
    else:
        webpage_name = 'Detect_Visualization_' + op.basename(args.folder) + '_' + args.webid + '_emis'
        match_catalog = 'continuum_matches_'+ op.basename(args.folder) + '_' + args.webid + '.dat'
        emis_catalog = 'emission_catalog_'+ op.basename(args.folder) + '_' + args.webid + '.dat'
    if args.create_header:
        webpage_name = 'Detect_Visualization_' + op.basename(args.folder)+'_header'+ '_emis'
    if args.create_ending:
        webpage_name = 'Detect_Visualization_' + op.basename(args.folder)+'_ending'+ '_emis'
    with open(match_catalog,'w') as f_match:
#        f_match.write('%s   %s   %s   %s   %s   %s   %s   %s\n'
#                      % ('Field'.ljust(9), 'SPECID'.ljust(9), ('RA_det').ljust(9), 
#                        ('Dec_det').ljust(8), 
#                        ('RA_cat').ljust(9),
#                        ('Dec_det').ljust(8),
#                        ('g-mag').ljust(8), 
#                        ('Dist').ljust(4)))
        f_match.flush()
    non_sortable_cols = [7,8,9,10]
    non_sortable_cols_cont = [6,7,8]
    with open(webpage_name+'.html', 'w') as f_webpage,\
         open(webpage_name[:-5]+'_cont.html', 'w') as f_cont_webpage,\
         open(emis_catalog,'w') as f_emis:
        if args.create_header:
            CW.CreateWebpage.writeHeader(f_webpage,webpage_name)
            CW.CreateWebpage.writeColumnNames(f_webpage,columnnames,non_sortable_cols)
            CW.CreateWebpage.writeHeader(f_cont_webpage,webpage_name[:-5]+'_cont')
            CW.CreateWebpage.writeColumnNames(f_cont_webpage,columnnames_cont,non_sortable_cols_cont) 
        elif args.create_ending:
            CW.CreateWebpage.writeEnding(f_webpage)     
            CW.CreateWebpage.writeEnding(f_cont_webpage)
        else:
            EmisCatalog.writeHeader(f_emis)
            fplane = FPlane(fplane_file)
            tp = TP(args.ra, args.dec, args.rot)
            if args.goodsn:
                image_fn='/work/03564/stevenf/maverick/GOODSN/gn_acs_old_f435w_060mas_v2_drz.fits'
                cat_fn='/work/03229/iwold/maverick/stackCOSMOS/cat_g.fits'                
            elif args.cosmos:
                image_fn='/work/03229/iwold/maverick/stackCOSMOS/nano/COSMOS_g_sci.fits'
                cat_fn='/work/03229/iwold/maverick/stackCOSMOS/cat_g.fits'
            else:
                image_fn, cat_fn = pick_image(args.ra, args.dec)
            wcs = WCS(image_fn)
            data = fits.open(image_fn)[0].data
            catalog = fits.open(cat_fn)[1].data
            if not op.exists('images'):
                os.mkdir('images')
            if args.webid is None:
                CW.CreateWebpage.writeHeader(f_webpage,webpage_name)
                CW.CreateWebpage.writeColumnNames(f_webpage,columnnames,non_sortable_cols)
                CW.CreateWebpage.writeHeader(f_cont_webpage,webpage_name[:-5]+'_cont')
                CW.CreateWebpage.writeColumnNames(f_cont_webpage,columnnames_cont,non_sortable_cols_cont)         
            for specid in args.specid:
                ifux = fplane.by_ifuslot(CAM_IFUSLOT_DICT[specid]).x
                ifuy = fplane.by_ifuslot(CAM_IFUSLOT_DICT[specid]).y
                if args.debug:
                    print(specid)
                ifu_fn = op.join(virus_config, 'IFUcen_files', 'IFUcen_VIFU' + CAM_IFU_DICT[specid] + '.txt')
                if not op.exists(ifu_fn):
                    ifu_fn = op.join(virus_config, 'IFUcen_files', 'IFUcen_HETDEX.txt')
                if args.debug:
                    print(ifu_fn)
                IFU = IFUCenter(ifu_fn)
                Di = ParseDither(op.join(args.folder, 'c'+specid, args.dither_file))
                D = {}
                for s in SIDE:
                    D[s] = Distortion(op.join(args.folder, 'c'+specid, 
                                         Di.deformer[0]+'_%s.dist' %s))
                detect_fn = op.join(args.folder, 'c'+specid, 'detect_line.dat')
                detect_cont_fn = op.join(args.folder, 'c'+specid, 'detect_cont.dat')
                if op.exists(detect_cont_fn):
                    Cat1 = np.loadtxt(detect_cont_fn, dtype={'names': ('ID', 'icx', 
                                                                 'icy', 'sigma', 'fwhm_xy', 
                                                                 'a', 'b', 
                                                                 'pa', 'ir1', 
                                                                 'ka', 'kb', 
                                                                 'xmin', 'xmax', 
                                                                 'ymin', ' ymax', 
                                                                 'zmin', 'zmax'),
                                                 'formats': ('i4', np.float, np.float,
                                                             np.float, np.float, np.float, np.float, 
                                                             np.float, np.float, np.float, np.float,
                                                             np.float, np.float, np.float, np.float,
                                                             np.float, np.float)}, ndmin=1)
                    if not Cat1.size:
                        if args.debug:
                            print("No continuum sources for specid %s" %specid)
                    else:
                        make_continuum_row(Cat1, f_cont_webpage, args, D, Di, ifux, 
                                           ifuy, IFU, tp, specid, wcs, data, catalog,
                                           op.basename(args.folder), match_catalog)
                    
                if op.exists(detect_fn):
                    Cat = np.loadtxt(detect_fn, dtype={'names': ('NR', 'ID', 'XS', 
                                                                 'YS', 'l', 'z', 
                                                                 'dataflux', 'modflux', 
                                                                 'fluxfrac', 'sigma', 
                                                                 'chi2', 'chi2s', 
                                                                 'chi2w', 'gammq', 
                                                                 'gammq_s', 'eqw', 
                                                                 'cont'),
                                                 'formats': ('i4', 'i4', np.float, np.float,
                                                             np.float, np.float, np.float, np.float, 
                                                             np.float, np.float, np.float, np.float,
                                                             np.float, np.float, np.float, np.float,
                                                             np.float)},ndmin=1)
                    if not Cat.size:
                        continue
                    
                    make_emission_row(Cat, f_webpage, args, D, Di, ifux, ifuy, 
                                      IFU, tp, specid, wcs, data, f_emis)
            if args.webid is None:
                CW.CreateWebpage.writeEnding(f_webpage)     
                CW.CreateWebpage.writeEnding(f_cont_webpage)      
if __name__ == '__main__':
    main() 
