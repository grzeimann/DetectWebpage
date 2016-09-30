"""

@author: gregz

"""

from __future__ import print_function
import argparse as ap
from pyhetdex.cure.distortion import Distortion
import pyhetdex.tools.files.file_tools as ft
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os.path as op
import os
import textwrap
import CreateWebpage as CW
from collections import OrderedDict
from pyhetdex.het.ifu_centers import IFUCenter
from astropy.stats import biweight_location, biweight_midvariance
from scipy.ndimage.filters import gaussian_filter
from astropy.modeling.models import Moffat2D, Gaussian2D
from photutils import CircularAperture, aperture_photometry
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from pyhetdex.het.fplane import FPlane
from pyhetdex.coordinates.tangent_projection_astropy import TangentPlane as TP


dist_thresh = 2. # Fiber Distance
sn_cut = 4.0 # S/N Cut
xw = 24 # image width in x-dir 
yw = 10 # image width in y-dir
res = [3,9]
ww = xw*1.9 # wavelength width
virus_config = '/work/03946/hetdex/maverick/virus_config'
fplane_file = '/home/00115/gebhardt/fplane.txt' 
image_dir = '/work/03229/iwold/maverick/fall_field/stack/v2/psf/nano'
#virus_config = '/Users/gregz/cure/virus_early/virus_config'

SPECBIG = ["L","R"]  

CAM_IFUSLOT_DICT = {'004':'093',
                    '037':'074',
                    '027':'075',                
                    '047':'076',
                    '024':'073',
                    '013':'084',
                    '016':'085',
                    '041':'086',
                    '051':'083',
                    '008':'094',
                    '025':'095',
                    '038':'096',
                    '020':'103',
                    '032':'104',
                    '012':'105',
                    '017':'106',}

IFUSLOT_DICT = {'073':['024','033'],
                '074':['037','024'],
                '075':['027','001'],
                '076':['047','016'],
                '083':['051','023'],
                '084':['013','019'],
                '085':['016','026'],
                '086':['041','015'],
                '093':['004','051'],
                '094':['008','054'],
                '095':['025','020'],
                '096':['038','014'],
                '103':['020','004'],
                '104':['032','028'],
                '105':['012','055'],
                '106':['017','022'],}

# Dictionary of the mapping between SPECID and IFUID

CAM_IFU_DICT = {'004':'051',
                '037':'024',
                '027':'001',                
                '047':'016',
                '024':'033',
                '013':'019',
                '016':'026',
                '041':'015',
                '051':'023',
                '008':'054',
                '025':'020',
                '038':'014',
                '020':'004',
                '032':'028',
                '012':'055',
                '017':'022',}
                
# Default set of spectrographs for reduction
SPECID = ["004","008","012","013","016","017","020","024","025","027","032",
          "037","038","041","047","051"]
SPECID = ["051"]
SIDE = ["L", "R"]

columnnames = ["SPECID", "NR", "ID", "Source_Info", "2D Plots","Spec Plots","Cutouts"]

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

class ParseDetect():
    """
    Parse the detect file from Karl's Output

    Parameters
    ----------
    detect_file : string
        file containing the detect output for a field.

    """

    def __init__(self, detect_file):
        self._absfname = op.abspath(detect_file)
        self.rah, self.dech= [], []
        self.ifuslot, self.sn = [], []
        self.chi2, self.wave, self.flux = [], [], []
        self.file, self.nr, self.id = [], [], []
        self._read_dectect(detect_file)

    def _read_detect(self, detect_file):
        """
        Read the detect file from Karl

        Parameters
        ----------
        detect_file : string
            file containing the detected sources
        """
        with open(detect_file, 'r') as f:
            f = ft.skip_comments(f)
            for l in f:
                try:
                    _rah, _dech, _ifuslot, _sn, _chi2, _wave, _flux, _file, _nr, _id = l.split()
                except ValueError:  # skip empty or incomplete lines
                    pass
                self.rah.append(float(_rah))
                self.dech.append(float(_dech))
                self.ifuslot.append(int(_ifuslot))
                self.sn.append(float(_sn))
                self.chi2.append(float(_chi2))
                self.wave.append(float(_wave))
                self.flux.append(float(_flux))
                self.file.append(_file.split(':')[0])
                self.nr.append(int(_nr))
                self.id.append(int(_id))

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

    parser.add_argument("--detect_file", nargs='?', type=str, 
                        help='''Detect File ''', 
                        default='detect.txt')           

    parser.add_argument("--ra", nargs='?', type=float, 
                        help='''ra''', 
                        default=None)
                        
    parser.add_argument("--dec", nargs='?', type=float, 
                        help='''Dec''', 
                        default=None)

    parser.add_argument("--rot", nargs='?', type=float, 
                        help='''rotation''', 
                        default=None)

    parser.add_argument("--debug", help='''Debug''',
                        action="count", default=0)   
                                                                         
    args = parser.parse_args(args=argv)

    # Check that the arguments are filled
    #if args.detect_file is not None:
    #    args.detect_file = args.detect_file.replace(" ", "").split(',')
    #else:
    #    msg = 'No detect file was provided'
    #    parser.error(msg)         

    # Check that the arguments are filled

    if args.ra is None:
        msg = 'No RA was provided'
        parser.error(msg)
    if args.dec is None:
        msg = 'No Dec was provided'
        parser.error(msg)
    if args.rot is None:
        msg = 'No Parangle was provided'
        parser.error(msg)
    if args.folder is None:
        msg = 'No folder was provided'
        parser.error(msg)
    if args.dither_file is None:
        msg = 'No dither file was provided'
        parser.error(msg)
    else:
        args.dither_file = args.dither_file
 
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
    return filename
    
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
    
def build_spec_image(datakeep, outfile, cwave, dwave=1.0, cmap=None, debug=False):
    if not cmap:
        # Default cmap is gray
        cmap = plt.get_cmap('gray_r')
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
    for i in xrange(N):
        specplot.plot(datakeep['specwave'][i], datakeep['spec'][i], 
                      alpha=0.5, color='r')
        w1 = np.interp(datakeep['d'][i],r,w)
        F+=(np.interp(bigwave,datakeep['specwave'][i], datakeep['spec'][i])*w1)
        W+=w1
        mn = np.min([mn,np.min(datakeep['spec'][i])])
        mx = np.max([mx,np.max(datakeep['spec'][i])])
    F /= W
    specplot.plot(bigwave, F, c='b')
    ran = mx - mn
    specplot.plot([cwave,cwave],[mn-ran*rm, mn+ran*(1+rm)])
    specplot.axis([cwave-ww, cwave+ww, mn-ran*rm, mn+ran*(1+rm)])
    fig.savefig(outfile,dpi=150)
    plt.close(fig)


def make_image_cutout(datakeep, data, wcs, ras, decs, outfile, cmap2=None,
                      cmap=None, size=50., debug=False):
    if not cmap:
        # Default cmap is gray
        cmap = plt.get_cmap('gray_r')
    if not cmap2:
        norm = plt.Normalize()
        colors = plt.cm.viridis(norm(np.arange(len(datakeep['ra'])+2)))
    pixsize_x = np.sqrt(wcs.wcs.cd[0,0]**2 + wcs.wcs.cd[0,1]**2)*3600. 
    pixsize_y = np.sqrt(wcs.wcs.cd[1,0]**2 + wcs.wcs.cd[1,1]**2)*3600. 
    sz = size * pixsize_x
    position = SkyCoord(ras, decs, unit="deg", frame='fk5')   
    cutout = Cutout2D(data, position, (size,size), wcs=wcs)
    fig = plt.figure(figsize=(4,4))
    plt.imshow(cutout.data,origin='lower',interpolation='nearest',vmin=-10,vmax=50, 
               cmap=cmap, extent=[-sz/2.,sz/2.,-sz/2.,sz/2.])
    xc, yc = skycoord_to_pixel(position, wcs=cutout.wcs)
    plt.scatter(0., 0.,marker='x',c='r',s=35)
    circle = plt.Circle((0., 0.), radius=2., fc='none', 
                            ec='r', zorder=2, alpha=1.0)
    plt.gca().add_patch(circle)
    for i in xrange(len(datakeep['ra'])):
        xf,yf = skycoord_to_pixel(
             SkyCoord(datakeep['ra'][i],datakeep['dec'][i], unit="deg", frame='fk5'), 
             wcs=cutout.wcs)
        circle = plt.Circle(((xf-xc)*pixsize_x, (yf-yc)*pixsize_x), radius=.75, fc='none', 
                            ec=colors[i,0:3], zorder=2, alpha=0.6)
        plt.gca().add_patch(circle)
    fig.savefig(outfile,dpi=150)
    plt.close(fig)        
    
def build_2d_image(datakeep, outfile, cmap=None, cmap2=None, debug=False):
    if not cmap:
        # Default cmap is gray
        cmap = plt.get_cmap('gray_r')
    if not cmap2:
        norm = plt.Normalize()
        colors = plt.cm.viridis(norm(np.arange(len(datakeep['ra'])+2)))
    N = len(datakeep['xi'])
    bordbuff = 0.01
    borderxl = 0.05
    borderxr = 0.15
    borderyb = 0.05
    borderyt = 0.15
    dx = (1. - borderxl - borderxr - 2*bordbuff) / 3.
    dy = (1. - borderyb - borderyt - 2*N*bordbuff) / N
    Y = (yw / dy) / (xw / dx) * 5.

    fig = plt.figure(figsize=(5,Y))

    ind = sorted(range(len(datakeep['d'])), key=lambda k: datakeep['d'][k], 
                 reverse=True)
    for i in xrange(N):
        borplot = plt.axes([borderxl+0.*dx+bordbuff/2., borderyb+i*dy+bordbuff/2., 3*dx+bordbuff, dy+bordbuff])
        implot = plt.axes([borderxl+2.*dx+bordbuff, borderyb+i*dy+bordbuff, dx, dy])
        errplot = plt.axes([borderxl+1.*dx+bordbuff, borderyb+i*dy+bordbuff, dx, dy])
        cosplot = plt.axes([borderxl+0.*dx+bordbuff, borderyb+i*dy+bordbuff, dx, dy])
        autoAxis = borplot.axis()
        rec = plt.Rectangle((autoAxis[0],autoAxis[2]),(autoAxis[1]-autoAxis[0]),
                            (autoAxis[3]-autoAxis[2]), fill=False, lw=3, 
                            color = colors[i,0:3], zorder=1)
        rec = borplot.add_patch(rec)
        borplot.set_xticks([])
        borplot.set_yticks([]) 
        borplot.axis('off')
        mn = biweight_location(datakeep['im'][ind[i]])
        st = np.sqrt(biweight_midvariance(datakeep['im'][ind[i]]))
        vmin = mn - 5*st
        vmax = mn + 5*st
        beta = 1.
        ext = list(np.hstack([datakeep['xl'][ind[i]],datakeep['xh'][ind[i]],
                              datakeep['yl'][ind[i]],datakeep['yh'][ind[i]]]))
        GF = gaussian_filter(datakeep['im'][ind[i]],(2,1))  
        implot.imshow(GF, 
                      origin="lower", cmap=cmap, 
                      interpolation="nearest",vmin=vmin,
                      vmax=vmax,
                      extent=ext)
        implot.scatter(datakeep['xi'][ind[i]],datakeep['yi'][ind[i]],
                       marker='x',c='r',s=10)
        implot.set_xticks([])
        implot.set_yticks([])
        implot.axis(ext)
        errplot.imshow(datakeep['err'][ind[i]], 
                      origin="lower", cmap=cmap, 
                      interpolation="nearest",vmin=vmin,vmax=vmax,
                      extent=ext)
        errplot.scatter(datakeep['xi'][ind[i]],datakeep['yi'][ind[i]],
                       marker='x',c='r',s=10)
        errplot.set_xticks([])
        errplot.set_yticks([])
        errplot.axis(ext)
        a = datakeep['cos'][ind[i]]
        a = np.ma.masked_where(a==0, a)
        cmap1 = cmap
        cmap1.set_bad(color=[0.2,1.0,0.23])
        cosplot.imshow(a, 
                      origin="lower",cmap=cmap1,
                      interpolation="nearest",vmin=vmin,vmax=vmax,
                      extent=ext)
        cosplot.scatter(datakeep['xi'][ind[i]],datakeep['yi'][ind[i]],
                       marker='x',c='r',s=10)
        cosplot.set_xticks([])
        cosplot.set_yticks([])
        cosplot.axis(ext)
        xi = datakeep['xi'][ind[i]]
        yi = datakeep['yi'][ind[i]]
        xl = int(np.round(xi-ext[0]-res[0]/2.))
        xh = int(np.round(xi-ext[0]+res[0]/2.))
        yl = int(np.round(yi-ext[2]-res[0]/2.))
        yh = int(np.round(yi-ext[2]+res[0]/2.))
        S = np.where(datakeep['err'][ind[i]][yl:yh,xl:xh]<0,0.,datakeep['im'][ind[i]][yl:yh,xl:xh]).sum()
        N = np.sqrt(np.where(datakeep['err'][ind[i]][yl:yh,xl:xh]<0,0.,datakeep['err'][ind[i]][yl:yh,xl:xh]**2).sum())
        sn = S/N
        implot.text(1.05,.55,'S/N = %0.2f' %(sn),
                    transform=implot.transAxes,fontsize=8,color='r',
                    verticalalignment='bottom', horizontalalignment='left')
        implot.text(1.05,.20,'D(") = %0.2f' %(datakeep['d'][ind[i]]),
                    transform=implot.transAxes,fontsize=8,color='r',
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

    
def main():
    args = parse_args()
    webpage_name = 'Detect Visualization_' + op.basename(args.folder)
    non_sortable_cols = [3,4]
    fplane = FPlane(fplane_file)
    tp = TP(args.ra, args.dec, args.rot)
    image_fn = pick_image(args.ra, args.dec)
    wcs = WCS(image_fn)
    data = fits.open(image_fn)[0].data
    if not op.exists('images'):
        os.mkdir('images')
    with open(webpage_name+'.html', 'w') as f_webpage:
        CW.CreateWebpage.writeHeader(f_webpage,webpage_name)
        CW.CreateWebpage.writeColumnNames(f_webpage,columnnames,non_sortable_cols)
        for specid in SPECID:
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
            D_L = Distortion(op.join(args.folder, 'c'+specid, 
                                     Di.deformer[0]+'_L.dist'))
            D_R = Distortion(op.join(args.folder, 'c'+specid, 
                                     Di.deformer[0]+'_R.dist'))
            D = {}
            D[SIDE[0]] = D_L
            D[SIDE[1]] = D_R
            detect_fn = op.join(args.folder, 'c'+specid, 'detect_line.dat')
            if op.exists(detect_fn):
                Cat = np.loadtxt(detect_fn, dtype={'names': ('NR', 'ID', 'XS', 
                                                             'YS', 'l', 'z', 
                                                             'dataflux', 'modflux', 
                                                             'fluxfrac', 'sigma', 
                                                             'chi2', 'chi2s', 
                                                             'chi2w', 'gammq', 
                                                             'gammq_s', ' eqw', 
                                                             'cont'),
                                             'formats': ('i4', 'i4', np.float, np.float,
                                                         np.float, np.float, np.float, np.float, 
                                                         np.float, np.float, np.float, np.float,
                                                         np.float, np.float, np.float, np.float,
                                                         np.float)})
                
                for i in xrange(len(Cat['XS'])):
                    x = Cat['XS'][i]
                    y = Cat['YS'][i]
                    sn = Cat['sigma'][i]
                    chi2 = Cat['chi2'][i]
                    flux = Cat['dataflux'][i]
                    datakeep = {}
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
                    datakeep['err'] = []
                    datakeep['spec'] = []
                    datakeep['specwave'] = []
                    datakeep['cos'] = []
                    datakeep['par'] = []
                    datakeep['ra'] = []
                    datakeep['dec'] = []
                    ras, decs = tp.xy2raDec(x+ifuy,y+ifux)
                    if sn>sn_cut:
                        for side in SIDE:
                            for dither in xrange(len(Di.dx)):
                                dx = x-IFU.xifu[side]-Di.dx[dither]
                                dy = y-IFU.yifu[side]-Di.dy[dither]
                                d = np.sqrt(dx**2 + dy**2)
                                loc = np.where(d<dist_thresh)[0]
                                for l in loc:
                                    f0 = D[side].get_reference_f(l+1)
                                    xi = D[side].map_wf_x(Cat['l'][i],f0)
                                    yi = D[side].map_wf_y(Cat['l'][i],f0)
                                    xfiber = IFU.xifu[side][l]+Di.dx[dither]
                                    yfiber = IFU.yifu[side][l]+Di.dy[dither]
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
                                    if args.debug:
                                        Di.basename[dither]
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
                                    if op.exists(im_fn):
                                        datakeep['im'].append(fits.open(im_fn)[0].data[yl:yh,xl:xh])
                                        datakeep['par'].append(fits.open(im_fn)[0].header['PARANGLE'])
                                    if op.exists(err_fn):
                                        datakeep['err'].append(fits.open(err_fn)[0].data[yl:yh,xl:xh])
                                    if op.exists(cos_fn):
                                        datakeep['cos'].append(fits.open(cos_fn)[0].data[yl:yh,xl:xh])
                                    if op.exists(FE_fn):
                                        FE = fits.open(FE_fn)[0].data
                                        nfib, xlen = FE.shape
                                        crval = fits.open(FE_fn)[0].header['CRVAL1']
                                        cdelt = fits.open(FE_fn)[0].header['CDELT1']
                                        wave = np.arange(xlen)*cdelt + crval
                                        Fe_indl = np.searchsorted(wave,Cat['l'][i]-ww,side='left')
                                        Fe_indh = np.searchsorted(wave,Cat['l'][i]+ww,side='right')
                                        datakeep['spec'].append(FE[l,Fe_indl:(Fe_indh+1)])
                                        datakeep['specwave'].append(wave[Fe_indl:(Fe_indh+1)])

                        outfile_2d = ('images/image2d_%s_specid_%s_object_%i_%i.png' 
                                    % (op.basename(args.folder), specid, Cat['NR'][i], 
                                       Cat['ID'][i]))
                        build_2d_image(datakeep, outfile_2d, debug=args.debug)
                        outfile_spec = ('images/imagespec_%s_specid_%s_object_%i_%i.png' 
                                    % (op.basename(args.folder), specid, Cat['NR'][i],
                                       Cat['ID'][i]))
                        build_spec_image(datakeep, outfile_spec, 
                                         cwave=Cat['l'][i], debug=args.debug)        
                        outfile_cut = ('images/imagecut_%s_specid_%s_object_%i_%i.png' 
                                    % (op.basename(args.folder), specid, Cat['NR'][i], 
                                       Cat['ID'][i]))
                        make_image_cutout(datakeep, data, wcs, ras, decs, 
                                          outfile_cut, debug=args.debug)
                        dict_web = OrderedDict()
                        dict_web['Number_1'] = int(specid)
                        dict_web['Number_2'] = int(Cat['NR'][i])
                        dict_web['Number_3'] = int(Cat['ID'][i])
                        dict_web['Table_1'] = [('S/N: %0.2f' %(sn)),
                                               ('chi2: %0.2f' %(chi2)),
                                               ('flux: %0.1f'% (flux))]
                        dict_web['Image_1'] = outfile_2d
                        dict_web['Image_2'] = outfile_spec
                        dict_web['Image_3'] = outfile_cut
                        CW.CreateWebpage.writeColumn(f_webpage,dict_web)  
        CW.CreateWebpage.writeEnding(f_webpage)     
       
    
if __name__ == '__main__':
    main() 
