"""
Example webpage creation.

.. moduleauthor:: Greg Zeimann <gregz@astro.as.utexas.edu>

"""

from astropy.io import fits
import numpy as np
import os.path as op
import CreateWebpage as CW
from collections import OrderedDict
import matplotlib.pyplot as plt
import os
import shutil
import argparse as ap 

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
    description = '''make_3dhst_webpage.py '''
                     
    parser = ap.ArgumentParser(description=description,
                            formatter_class=ap.RawTextHelpFormatter)
                            
    parser.add_argument("-fd","--field", type=str, nargs='?',
                        help='''AEGIS, GOODSN, COSMOS''',
                        default=None)

    parser.add_argument("-fn","--filename", type=str, nargs='?',
                        help='''sources.dat''',
                        default=None)
    
    parser.add_argument("-r","--rootdir", type=str, nargs='?',
                        help='''/FOLDER''',
                        default="/Users/gregz/")
                    
    args = parser.parse_args(args=argv)
    
    if args.field is None:
        msg = 'Please fill in the field'
        parser.error(msg)
    else:
        args.field = args.field.upper()
    if args.filename is None:
        msg = 'Please fill in the filename'
        parser.error(msg)
        
    return args

def fits2png(filename, ext, low, high, cmap=None):
    '''Convert a 2d fits image into an png image
    
    Parameters
    ----------
    filename: fits filename
    ext: extension for the fits image to convert
    cmap: colormap
    '''
    if not cmap:
        # Default cmap is gray
        cmap = plt.get_cmap('gray_r')
    p = fits.open(filename)
    image = p[ext].data
    A,B = image.shape
    pixsize = 0.012
    fig = plt.figure(figsize=(B*pixsize,A*pixsize))
    beta = 1
    plt.imshow(np.arcsinh(image/beta), origin="lower", cmap=cmap, 
               interpolation="nearest", 
               vmin=np.arcsinh(low/beta), vmax=np.arcsinh(high/beta))
    plt.xticks([])
    plt.yticks([])
    fig.savefig(filename[:-5]+'.png',dpi=150)
    plt.close(fig)
    
def publish(filename, folder):
    if not op.exists(folder):
        os.mkdir(folder)
    shutil.copy(filename,op.join(folder,op.basename(filename)))
    return op.join(folder,op.basename(filename))

def build_filename(args, catalog):
    return op.join(args.rootdir, '%s_WFC3_V4.1.5' %args.field, 
                   '%s_3dhst_v4.1.5_catalogs' %args.field.lower(),
                   '%s_3dhst.v4.1.%s' %(args.field.lower(), catalog))

def main():
    args = parse_args()    
    columnnames = ["Number","Phot ID", "Location", "JH_mag", "z_3dhst_best", 
                   "VIRUS_Redshift", "H-alpha", 
                   "H-beta", "Image", "2D Spectrum", "Fit"]

    base_location = op.join(args.rootdir, 'GOODSN_WFC3_V4.1.5')
    catalog = fits.open(build_filename(args,'5.zfit.linematched.fits'))
    phot_catalog = fits.open(build_filename(args, 'cat.FITS'))
    ra,dec,wave = np.loadtxt(args.filename, usecols=(0,1,2), unpack=True)
    thresh_d = 3.
    rkeep = []
    dkeep = []
    drkeep = []
    ddkeep = []
    dzkeep = []
    zkeep = []
    idkeep = []
    nidkeep = []
    if isinstance(ra, float):
        cnt = 1
        ra = [ra]
        dec = [dec]
        wave = [wave]
    else:
        cnt = len(ra)
    flag=np.zeros((cnt,),dtype=bool)
    for i in xrange(cnt):
        dra = ((ra[i] - phot_catalog[1].data['ra'])
                * np.cos(np.pi/180.*dec[i]) * 3600.)
        ddec = (dec[i] - phot_catalog[1].data['dec'])*3600.
        d = np.sqrt(dra**2 + ddec**2)

        seld = np.where(d < thresh_d)[0]
        if len(seld):
            flag[i]=True
        for l in seld:
            drkeep.append(dra[l])
            ddkeep.append(ddec[l])
            idkeep.append(l)
            nidkeep.append(i)
            rkeep.append(phot_catalog[1].data['ra'][l])
            dkeep.append(phot_catalog[1].data['dec'][l])
            zkeep.append(catalog[1].data['z_best'][l])

    rkeep = np.array(rkeep)
    dkeep = np.array(dkeep)
    zkeep = np.array(zkeep)
    drkeep = np.array(drkeep)
    ddkeep = np.array(ddkeep)
    dzkeep = np.array(dzkeep)
    idkeep = np.hstack(np.array(idkeep))
    nidkeep = np.hstack(np.array(nidkeep))


    N = len(idkeep)
    idx = idkeep
    webpage_name = '3D-HST %s, VIRUS matches' %args.field
    non_sortable_cols = [2,7,8,9]

    with open(webpage_name+'.html', 'w') as f_webpage:
        CW.CreateWebpage.writeHeader(f_webpage,webpage_name)
        CW.CreateWebpage.writeColumnNames(f_webpage,columnnames,non_sortable_cols)
        for i in xrange(N):
            dict_web = OrderedDict()
            dict_web['Number_1'] = nidkeep[i]+1
            dict_web['Number_2'] = catalog[1].data['phot_id'][idx[i]]
            dict_web['Table_1'] = [('RA: %0.5f' 
                                    %(phot_catalog[1].data['ra'][idx[i]])),
                                   ('Dec: %0.5f'
                                    % (phot_catalog[1].data['dec'][idx[i]]))]
            dict_web['Number_4'] = catalog[1].data['jh_mag'][idx[i]]
            dict_web['Number_5'] = catalog[1].data['z_best'][idx[i]]
            dict_web['Table_2'] = [('Virus Wave: %0.1f' 
                                    %(wave[nidkeep[i]])),
                                   ('OII_z: %0.3f' 
                                    %(wave[nidkeep[i]]/3727. - 1.)),
                                   ('LAE_z: %0.3f'
                                    % (wave[nidkeep[i]]/1216. - 1.))]
            grism_id = catalog[1].data['grism_id'][idx[i]]
            if grism_id is '00000':
                folder = op.join(base_location, 
                                 grism_id.split('-')[0] + '-' + grism_id.split('-')[1])
                filename = op.join(folder, '2D', 'FITS',
                                              grism_id+'.2D.fits') 
                fits2png(filename, 1, -0.03, 0.5)
                new_location_1 = publish(op.join(folder, '2D', 'FITS', 
                                                 grism_id+'.2D.png'), 'images')
                new_location_2 = publish(op.join(folder, 'ZFIT', 'PNG', 
                                                 grism_id+'.new_zfit.2D.png'), 'images')
                new_location_3 = publish(op.join(folder, 'ZFIT', 'PNG', 
                                                 grism_id+'.new_zfit.png'), 'images')
                dict_web['Image_1'] = new_location_1
                dict_web['Image_2'] = new_location_2
                dict_web['Image_3'] = new_location_3   

            CW.CreateWebpage.writeColumn(f_webpage,dict_web)
        CW.CreateWebpage.writeEnding(f_webpage)

if __name__=='__main__':
    main()