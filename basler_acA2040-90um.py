# Third party imports, installable via pip:
import numpy as np
from pypylon import pylon

class Camera:
    '''
    Basic device adaptor for Basler acA2040-90um USB 3.0 camera. Many more
    commands are available and have not been implemented.
    '''
    def __init__(self,
                 name='acA2040-90um',
                 cameras=1,
                 verbose=True,
                 very_verbose=False):
        self.name = name
        assert cameras == 1, "%s: currently only 1 camera supported"%self.name
        self.verbose = verbose
        self.very_verbose = very_verbose
        if self.verbose: print("%s: opening..."%self.name)
        # init:
        self._get_device_count()
        if self.device_count == 0:
            print("%s: -> is the camera on and connected?"%self.name)
            raise Exception("%s: no devices found;"%self.name)
        if self.device_count > 1:
            print("%s: -> only 1 camera supported (found > 1)"%self.name)
            raise Exception("%s: too many devices found;"%self.name)
        try:
            self.device = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateFirstDevice())
        except:
            print("%s: -> is pylon Viewer running? (close it)"%self.name)
            print("%s: -> is another Python shell open? (close it)"%self.name)
            raise Exception("%s: failed to open;"%self.name)
        self.device.Open()
        # get device info and set defaults:
        self.px_um = 5.5
        self._get_model_name()
        # turn off fps and bandwidth (bytes per second) limiters:
        self.device.AcquisitionFrameRateEnable.Value = False
        self.device.DeviceLinkThroughputLimitMode.Value = 'Off'
        self.device.ExposureAuto.Value = "Off"      # turn off auto exposure
        self._get_exposure_time_us()
        self.device.GainAuto.Value = "Off"          # turn off auto gain
        self._get_gain()
        self.device.PixelFormat.Value = "Mono12"    # support 16bit only
        self._get_roi()                             # + readout_time_us
        self.apply_settings(trigger='software',     # default software trigger
                            num_buffers=10)         # default 10 buffers?
        if self.verbose: print("%s: -> open and ready."%self.name)

    def _get_device_count(self):
        if self.very_verbose:
            print("%s: getting device count"%self.name, end='')
        self.device_count = len(
            pylon.TlFactory.GetInstance().EnumerateDevices())
        if self.very_verbose:
            print(" = %s"%self.device_count)
        return self.device_count

    def _get_model_name(self):
        if self.very_verbose:
            print("%s: getting model name"%self.name, end='')
        self.model_name = self.device.GetDeviceInfo().GetModelName()
        if self.very_verbose:
            print(" = %s"%self.model_name)
        return        

    def _get_exposure_time_us(self):
        if self.very_verbose:
            print("%s: getting exposure time (us)"%self.name, end='')
        self.exposure_us = self.device.ExposureTime.Value
        if self.very_verbose:
            print(" = %08i"%self.exposure_us)
        return self.exposure_us

    def _set_exposure_time_us(self, exposure_us):
        if self.very_verbose:
            print("%s: setting exposure time (us) = %08i"%(
                self.name, exposure_us))
        assert type(exposure_us) is int, (
            "%s: unexpected type for exposure_us"%self.name)
        assert 28 <= exposure_us <= 1e7, (
            "%s: exposure_us out of range"%self.name)
        self.device.ExposureTime.Value = exposure_us
        assert self._get_exposure_time_us() == exposure_us
        if self.very_verbose:
            print("%s: -> done setting exposure time."%self.name)
        return None

    def _get_gain(self):
        if self.very_verbose:
            print("%s: getting gain"%self.name, end='')
        self.gain = self.device.Gain.Value
        if self.very_verbose:
            print(" = %02f"%self.gain)
        return self.gain

    def _set_gain(self, gain):
        if self.very_verbose:
            print("%s: setting gain = %02f"%(self.name, gain))
        assert type(gain) is int or type(gain) is float, (
            "%s: unexpected type for exposure_us"%self.name)
        assert 0 <= gain <= 23, ("%s: gain out of range"%self.name)
        self.device.Gain.Value = gain
        assert gain - 0.15 <= self._get_gain() <= gain + 0.15 # 0.15 tolerance
        if self.very_verbose:
            print("%s: -> done setting gain."%self.name)
        return None

    def _get_roi(self):
        if self.very_verbose:
            print("%s: getting roi pixels"%self.name, end='')
        self.height_px = self.device.Height.Value
        self.width_px =  self.device.Width.Value
        self.readout_time_us = self.device.SensorReadoutTime.Value
        if self.very_verbose:
            print(" = %i x %i (height x width)"%(self.height_px, self.width_px))
        return self.height_px, self.width_px

    def _set_roi(self, height_px, width_px):
        # only certain values and combinations allowed -> use legalizer!
        if self.very_verbose:
            print("%s: setting roi = %i x %i (height x width)"%(
                self.name, height_px, width_px))
        self.device.Height.Value = height_px
        self.device.Width.Value  = width_px
        assert self._get_roi() == (height_px, width_px)
        if self.very_verbose:
            print("%s: -> done setting roi."%self.name)
        return None

    def _get_trigger_setup(self):
        if self.very_verbose:
            print("%s: getting trigger setup:"%self.name)
        self.trigger_selector   = self.device.TriggerSelector.Value
        self.trigger_mode       = self.device.TriggerMode.Value
        self.trigger_activation = self.device.TriggerActivation.Value
        self.trigger_source     = self.device.TriggerSource.Value
        if self.very_verbose:
            print("%s: -> trigger_selector   = %s"%(
                self.name, self.trigger_selector))
            print("%s: -> trigger_mode       = %s"%(
                self.name, self.trigger_mode))
            print("%s: -> trigger_activation = %s"%(
                self.name, self.trigger_activation))
            print("%s: -> trigger_source     = %s"%(
                self.name, self.trigger_source))
        return (self.trigger_selector,
                self.trigger_mode,
                self.trigger_activation,
                self.trigger_source)

    def _set_trigger_setup(self,
                           trigger_selector,
                           trigger_mode,
                           trigger_activation,
                           trigger_source):
        if self.very_verbose:
            print("%s: setting trigger setup:"%self.name)
            print("%s: -> trigger_selector   = %s"%(
                self.name, trigger_selector))
            print("%s: -> trigger_mode       = %s"%(
                self.name, trigger_mode))
            print("%s: -> trigger_activation = %s"%(
                self.name, trigger_activation))
            print("%s: -> trigger_source     = %s"%(
                self.name, trigger_source))
        assert trigger_selector in ('FrameStart',), (
            "%s: trigger_selector (%s) unavailable"%(
                self.name, trigger_selector))
        assert trigger_mode in ('On', 'Off'), (
            "%s: trigger_mode (%s) unavailable"%(
                self.name, trigger_mode))
        assert trigger_activation in ('RisingEdge',), (
            "%s: trigger_activation (%s) unavailable"%(
                self.name, trigger_activation))
        assert trigger_source in ('Line1', 'Software'), (
            "%s: trigger_source (%s) unavailable"%(
                self.name, trigger_source))
        self.device.TriggerSelector.Value   = trigger_selector
        self.device.TriggerMode.Value       = trigger_mode
        self.device.TriggerActivation.Value = trigger_activation
        self.device.TriggerSource.Value     = trigger_source
        assert self._get_trigger_setup() == (
            trigger_selector, trigger_mode, trigger_activation, trigger_source)
        if self.very_verbose:
            print("%s: -> done setting trigger setup."%self.name)
        return None

    def _get_num_buffers(self):
        if self.very_verbose:
            print("%s: getting num buffers"%self.name, end='')
        self.num_buffers = self.device.MaxNumBuffer.Value
        if self.very_verbose:
            print(" = %i"%self.num_buffers)
        return self.num_buffers

    def _set_num_buffers(self, num_buffers):
        if self.very_verbose:
            print("%s: setting num buffers = %i"%(self.name, num_buffers))
        assert type(num_buffers) is int, (
            "%s: unexpected type for num_buffers"%self.name)
        assert 1 <= num_buffers <= 4294967295, (
            "%s: num_buffers out of range"%self.name)
        self.device.MaxNumBuffer.Value = num_buffers
        assert self._get_num_buffers() == num_buffers
        if self.very_verbose:
            print("%s: -> done setting num buffers."%self.name)
        return None

    def apply_settings(
        self,
        num_images=None,    # total number of images to record, type(int)
        exposure_us=None,   # 28 <= type(int) <= 10,000,000
        gain=None,          # 0 <= type(float) <= 23
        height_px=None,     # adjusted by legalize_image_size(), type(int)
        width_px=None,      # adjusted by legalize_image_size(), type(int)
        trigger=None,       # "auto"/"software"/"external"
        num_buffers=None,   # 1 <= type(int) <= 16
        timeout_ms=None,    # buffer timeout, type(int)
        ):
        if self.verbose:
            print("%s: applying settings..."%self.name)
        if num_images is not None:
            assert isinstance(num_images, int), (
            "%s: unexpected type for num_images"%self.name)
            self.num_images = num_images
        if exposure_us is not None:
            self._set_exposure_time_us(exposure_us)
        if gain is not None:
            self._set_gain(gain)
        if height_px is not None or width_px is not None:
            if height_px is None: height_px = self.height_px
            if width_px  is None: width_px  = self.width_px
            height_px, width_px = legalize_image_size(
                height_px, width_px, name=self.name, verbose=self.verbose)
            self._set_roi(height_px, width_px)
        if trigger is not None:
            assert trigger in ('auto', 'software', 'external'),(
                "%s: unexpected trigger mode (%s)"%(self.name, trigger))
            if trigger == 'auto':
                self._set_trigger_setup(
                    'FrameStart', 'Off', 'RisingEdge', 'Software')
            if trigger == 'software':
                self._set_trigger_setup(
                    'FrameStart', 'On', 'RisingEdge', 'Software')                
            if trigger == 'external':
                self._set_trigger_setup(
                    'FrameStart', 'On', 'RisingEdge', 'Line1')
            self.trigger = trigger
        if num_buffers is not None:
            self._set_num_buffers(num_buffers)
        if timeout_ms is not None:
            assert type(timeout_ms) is int, (
            "%s: unexpected type for timeout_ms"%self.name)
            self.timeout_ms = timeout_ms
        else:
            self.timeout_ms = 20 + int(
                1e-3 * (self.readout_time_us + self.exposure_us))
        self.expected_fps = self.device.ResultingFrameRate.Value
        if self.verbose:
            print("%s: -> done applying settings."%self.name)
        return None

    def record_to_memory(
        self,
        allocated_memory=None,  # optionally pass numpy array for images
        software_trigger=True,  # False -> external trigger needed
        ):
        if self.verbose:
            print("%s: recording to memory..."%self.name)
        h_px, w_px = self.height_px, self.width_px
        if allocated_memory is None: # make numpy array if none given
            allocated_memory = np.zeros((self.num_images, h_px, w_px), 'uint16')
            output = allocated_memory # no memory provided so return some images
        else: # images placed in provided array
            assert isinstance(allocated_memory, np.ndarray), (
            "%s: unexpected type for allocated_memory"%self.name)
            assert allocated_memory.dtype == np.uint16, (
            "%s: unexpected dtype for allocated_memory"%self.name)
            assert allocated_memory.shape == (self.num_images, h_px, w_px), (
            "%s: unexpected shape for allocated_memory"%self.name)
            output = None # avoid returning potentially large array
        if self.trigger == 'auto':
            self.device.StartGrabbingMax(self.num_images)
        else: # trigger is 'software' or 'external'
            self.device.StartGrabbing(pylon.GrabStrategy_OneByOne)
        for i in range(self.num_images):
            if not self.device.IsGrabbing():
                raise Exception("%s: camera not grabbing"%self.name)
            if software_trigger:
                if self.device.WaitForFrameTriggerReady(
                    self.timeout_ms,
                    pylon.TimeoutHandling_ThrowException):
                    self.device.ExecuteSoftwareTrigger()
            image = self.device.RetrieveResult( # timeoutMs, TimeoutHandling
                self.timeout_ms, pylon.TimeoutHandling_ThrowException)
            if not image.GrabSucceeded():
                print("%s: -> error code: %s"%(self.name, image.GetErrorCode()))
                print("%s: -> error description: %s"%(
                    self.name, image.GetErrorDescription()))
                raise Exception("%s: image grab failed -> no image"%self.name)                
            allocated_memory[i, :, :] = image.Array # get image
            image.Release()
        self.device.StopGrabbing()
        if self.verbose:
            print("%s: -> done recording to memory."%self.name)
        return output

    def close(self):
        if self.verbose: print("%s: closing..."%self.name, end='')
        self.device.Close()
        if self.verbose: print(" done.")
        return None

def legalize_image_size(
    height_px='max', width_px='max', name='acA2040-90um', verbose=True):
    """returns a nearby legal image size at the *center* of the camera chip"""
    height_step, width_step = 1, 8 # device.Height.Inc, device.Width.Inc
    min_height, min_width, max_height, max_width = 1, 8, 2048, 2048
    ud_center = (max_height / 2)
    lr_center = (max_width  / 2)
    if verbose:
        print("%s: requested image size (pixels)"%name)
        print("%s:  = %s x %s (height x width)"%(name, height_px, width_px))
    if height_px == 'min': height_px = min_height
    if height_px == 'max': height_px = max_height        
    if width_px  == 'min': width_px  = min_width
    if width_px  == 'max': width_px  = max_width
    assert type(height_px) is int, (
        "%s: unexpected type for height_px"%name)
    assert type(width_px) is int, (
        "%s: unexpected type for width_px"%name)
    assert min_height <= height_px <= max_height, (
        "%s: height_px out of range"%name)
    assert min_width  <= width_px  <= max_width, (
        "%s: width_px out of range"%name)
    num_height_steps = (height_px // height_step)
    num_width_steps  = (width_px  // width_step)
    if num_height_steps % 2 != 0: num_height_steps += 1 # must be even for chip
    if num_width_steps  % 2 != 0: num_width_steps  += 1 # must be even for chip 
    height_px = height_step * num_height_steps # now legalized
    width_px  = width_step  * num_width_steps  # now legalized
    if verbose:
        print("%s: legal image size (pixels)"%name)
        print("%s:  = %i x %i (height x width)"%(name, height_px, width_px))
    return height_px, width_px

if __name__ == '__main__':
    import time
    from tifffile import imread, imwrite
    camera = Camera(verbose=True, very_verbose=True)

    print('\nTake some pictures:')
    camera.apply_settings(
        num_images=10, exposure_us=100, gain=0, height_px='min', width_px=100)
    images = camera.record_to_memory()
    imwrite('test0.tif', images, imagej=True)

    # Note: fps is ~2x in "Mono8" vs "Mono12" mode (90fps spec is for "Mono8")
    print('\nMax fps test:')
    frames = 1000
    camera.apply_settings(frames, 28, 0, 'min', 'max', 'auto')
    images = np.zeros(
        (camera.num_images, camera.height_px, camera.width_px), 'uint16')
    t0 = time.perf_counter()
    camera.record_to_memory(allocated_memory=images, software_trigger=False)
    time_s = time.perf_counter() - t0
    print("\nMax fps = %0.2f (expected=%0.2f)\n"%( # ~ 36.2 -> 4607 typical
        frames/time_s, camera.expected_fps))
    imwrite('test1.tif', images, imagej=True)

    print('\nMax fps test -> multiple recordings:')
    iterations = 10
    frames = 100
    camera.apply_settings(frames, 28, 0, 'min', 'max', 'auto')
    images = np.zeros(
        (camera.num_images, camera.height_px, camera.width_px), 'uint16')
    t0 = time.perf_counter()
    for i in range(iterations):
        camera.record_to_memory(
            allocated_memory=images, software_trigger=False)
    time_s = time.perf_counter() - t0
    total_frames = iterations * frames
    print("\nMax fps = %0.2f (expected=%0.2f)\n"%( # ~ 36.2 -> 4078 typical
        total_frames/time_s, camera.expected_fps))
    imwrite('test2.tif', images, imagej=True)

    print('\nRandom input testing:')
    num_acquisitions = 10 # tested to 10000
    min_h_px, min_w_px = legalize_image_size('min','min')
    max_h_px, max_w_px = legalize_image_size('max','max')
    camera.verbose, camera.very_verbose = False, False
    blank_frames, total_latency_ms = 0, 0
    for i in range(num_acquisitions):
        print('\nRandom input test: %06i'%i)
        num_img = np.random.randint(1, 10)
        exp_us  = np.random.randint(28, 100000)
        gain    = np.random.randint(0, 23)
        h_px    = np.random.randint(min_h_px, max_h_px)
        w_px    = np.random.randint(min_w_px, max_w_px)
        num_buf = np.random.randint(1, 10)
        camera.apply_settings(
            num_img, exp_us, gain, h_px, w_px, 'software', num_buf)
        images = np.zeros(
            (camera.num_images, camera.height_px, camera.width_px), 'uint16')
        t0 = time.perf_counter()
        camera.record_to_memory(allocated_memory=images)
        t1 = time.perf_counter()
        time_per_image_ms = 1e3 * (t1 - t0) / num_img
        latency_ms = time_per_image_ms - 1e-3 * camera.exposure_us
        total_latency_ms += latency_ms
        print("latency (ms) = %0.6f"%latency_ms)
        print("shape of images:", images.shape)
        if i == 0: imwrite('test3.tif', images, imagej=True)
        images = images[:,:,:]
        print("min image values: %s"%images.min(axis=(1, 2)))
        print("max image values: %s"%images.max(axis=(1, 2)))
        n_blank = num_img - np.count_nonzero(images.max(axis=(1, 2)))
        if n_blank > 0:
            blank_frames += n_blank
            print('%d blank frames received...'%n_blank)
    average_latency_ms = total_latency_ms / num_acquisitions
    print("\n -> total blank frames received = %i"%blank_frames)
    print(" -> average latency (ms) = %0.6f"%average_latency_ms)

    camera.close()
