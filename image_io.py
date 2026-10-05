"""Read calibrated channel stacks and explicitly mapped RGB TIFF exports."""
import numpy as np
import tifffile
import xml.etree.ElementTree as ET


RGB_COMPONENTS={'red':0,'green':1,'blue':2}
LENGTH_TO_UM={'m':1e6,'cm':1e4,'mm':1e3,'µm':1.,'μm':1.,'um':1.,'nm':1e-3,'pm':1e-6}


def validate_rgb_channel_order(order):
    if order is None:return None
    if (not isinstance(order,list) or len(order)!=3
            or any(not isinstance(color,str) for color in order)
            or set(order)!=set(RGB_COMPONENTS)):
        raise ValueError('rgb_channel_order must contain red, green, and blue exactly once')
    return list(order)


def validate_pixel_size(size):
    if size is None:return None
    if (not isinstance(size,list) or len(size)!=2
            or any(type(value) not in (int,float) for value in size)
            or not np.isfinite(size).all() or min(size)<=0):
        raise ValueError('pixel_size_um must contain two finite, positive numbers: X and Y in µm per pixel')
    return [float(value) for value in size]


def _layout(tf,parameters):
    page=tf.pages[0]
    order=validate_rgb_channel_order((parameters or {}).get('rgb_channel_order'))
    if page.photometric==tifffile.PHOTOMETRIC.RGB:
        if order is None:
            raise ValueError('RGB TIFF export: specify --rgb-channel-order, for example '
                             'blue green red for blue nuclei, green C2, and red C3')
        if len(tf.pages)!=1 or page.axes not in {'YXS','SYX'} or page.samplesperpixel!=3:
            raise ValueError('Expected a single 2D RGB image with exactly three color components')
        shape=(3,*[size for axis,size in zip(page.axes,page.shape) if axis!='S'])
        return shape,page.axes,list(page.shape),'rgb_export',order
    series=tf.series[0]
    if series.axes!='CYX' or len(series.shape)!=3:
        raise ValueError(f'Expected a multichannel 2D CYX image; found axes {series.axes} and shape {series.shape}')
    return series.shape,series.axes,list(series.shape),'channel_stack',None


def _ome_calibration(tf):
    if not tf.ome_metadata:return None
    try:
        root=ET.fromstring(tf.ome_metadata)
    except ET.ParseError as exc:
        raise ValueError('Invalid OME XML; cannot read spatial calibration') from exc
    # The reader uses the first image series; never borrow another image's scale.
    pixels=root.find('{*}Image/{*}Pixels')
    if pixels is None:return None
    values=[pixels.get('PhysicalSizeX'),pixels.get('PhysicalSizeY')]
    if values==[None,None]:return None
    sizes=[]
    for axis,value in zip('XY',values):
        if value is None:raise ValueError(f'OME metadata is missing PhysicalSize{axis}')
        # The OME schema defaults omitted physical-size units to micrometers.
        unit=pixels.get(f'PhysicalSize{axis}Unit','µm')
        if unit not in LENGTH_TO_UM:
            raise ValueError(f'Unsupported OME PhysicalSize{axis}Unit: {unit!r}')
        try:
            sizes.append(float(value)*LENGTH_TO_UM[unit])
        except ValueError as exc:
            raise ValueError(f'Invalid OME PhysicalSize{axis}: {value!r}') from exc
    sizes=validate_pixel_size(sizes)
    return *sizes,'OME PhysicalSizeX/PhysicalSizeY converted to µm'


def _calibration(tf,parameters=None):
    override=validate_pixel_size((parameters or {}).get('pixel_size_um'))
    if override is not None:
        return *override,'Explicit pixel_size_um override (µm per pixel)'
    ome=_ome_calibration(tf)
    if ome is not None:return ome
    ij=tf.imagej_metadata or {}
    tags=tf.pages[0].tags
    if ij.get('unit') in {'micron','um','µm','μm'}:
        microns_per_unit=1.
        source='ImageJ micron unit and TIFF resolution'
    elif tags.get('ResolutionUnit') is not None and tags['ResolutionUnit'].value in {2,3}:
        microns_per_unit=25400. if tags['ResolutionUnit'].value==2 else 10000.
        source='TIFF physical resolution (inch)' if tags['ResolutionUnit'].value==2 else 'TIFF physical resolution (centimeter)'
    else:
        raise ValueError('Unknown spatial calibration: no OME pixel sizes, ImageJ micron calibration, '
                         'or TIFF physical resolution; supply --pixel-size-um X Y from microscope calibration')
    try:
        xnum,xden=tags['XResolution'].value
        ynum,yden=tags['YResolution'].value
        sx=microns_per_unit*xden/xnum;sy=microns_per_unit*yden/ynum
    except (KeyError,ZeroDivisionError,TypeError,ValueError) as exc:
        raise ValueError('Missing or invalid TIFF pixel resolution') from exc
    if not np.isfinite([sx,sy]).all() or min(sx,sy)<=0:
        raise ValueError('Pixel sizes must be finite and positive')
    return sx,sy,source


def inspect_image(path,parameters=None,require_calibration=True):
    """Inspect layout without loading pixels, including EVOS RGB page metadata."""
    with tifffile.TiffFile(path) as tf:
        shape,axes,source_shape,kind,order=_layout(tf,parameters)
        dtype=tf.pages[0].dtype
        ij=dict(tf.imagej_metadata or {});ij.pop('LUTs',None)
        md={'axes':'CYX','shape':list(shape),'dtype':str(dtype),'bit_depth':dtype.itemsize*8,
            'imagej_metadata':ij,'page_count':len(tf.pages),'source_axes':axes,
            'source_shape':source_shape,'input_format':kind,'rgb_channel_order':order,
            'saturation_value':int(np.iinfo(dtype).max) if np.issubdtype(dtype,np.integer) else None}
        if require_calibration:
            sx,sy,source=_calibration(tf,parameters)
            md.update(pixel_size_x_um=sx,pixel_size_y_um=sy,calibration_source=source)
    return md


def read_channels(path,parameters=None):
    """Return CYX pixels in their native dtype; RGB components are only reordered."""
    order=validate_rgb_channel_order((parameters or {}).get('rgb_channel_order'))
    if order is None:return tifffile.imread(path)
    with tifffile.TiffFile(path) as tf:
        _,axes,_,kind,_=_layout(tf,parameters)
        if kind!='rgb_export':return tf.asarray()
        raw=tf.pages[0].asarray()
        return np.moveaxis(raw,axes.index('S'),0)[[RGB_COMPONENTS[color] for color in order]]


def read_image(path,parameters=None):
    return read_channels(path,parameters),inspect_image(path,parameters)
