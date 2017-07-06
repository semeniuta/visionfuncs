
from PIL import Image
from glob import glob
from itertools import cycle

from epypes.node import Node
from epypes.pipeline import Pipeline, SourcePipeline
from epypes import util

from flexvi.core.images import open_image

def get_filenames_generator(mask, in_cycle=False):
    files = glob(mask)
    gen = cycle(files) if in_cycle else iter(files)
    return gen

class FilenamesEventSource(SourcePipeline):
    def __init__(self, qout, mask, in_cycle=False):
        self._gen = get_filenames_generator(mask, in_cycle)

        def func():
            try:
                return next(self._gen)
            except StopIteration:
                return None

        name = util.create_name_with_uuid(self.__class__)
        node = Node(util.create_name_with_uuid(Node), func)
        SourcePipeline.__init__(self, name, [node], qout)

class GetImageFromDiskNode(Node):

    def __init__(self, gray=False):
        name = util.reate_name_with_uuid(self.__class__)
        Node.__init__(self, name, open_image, gray=gray)

class BatchImagesetReaderNode(Node):

    def __init__(self, mask, gray=False):

        name = util.create_name_with_uuid(self.__class__)
        Node.__init__(self, name, self._read_all_images, mask=mask, gray=gray)

    def _read_all_images(self, mask, gray):
        filenames = glob(mask)
        images = [open_image(fname, gray=gray) for fname in filenames]
        return images

if __name__ == '__main__':

    #fname_q = util.create_basic_queue()
    #src = FilenamesEventSource(fname_q, '/Users/alex/Dropbox/PhD/CODE/DATA/IMG/calib/amct/*.tif')

    rn = BatchImagesetReaderNode('/Users/alex/Dropbox/PhD/CODE/DATA/IMG/calib/amct/*.tif')
