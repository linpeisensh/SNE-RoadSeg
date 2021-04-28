from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./testresults/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test')
        parser.add_argument('--sequence', type=str, default='00', help='')
        # parser.add_argument('--data_path', type=str, default='/storage/remote/atcremers17/linp/dataset/kittic/sequences', help='path to frames')
        parser.add_argument('--data_path', type=str, default='/storage/remote/atcremers54/linp/kittic/sequences', help='path to frames')
        parser.add_argument('--depth_path', type=str, default='../depth', help='path to depth')
        # parser.add_argument('--save_path', type=str, default='../prob', help='path to save')
        parser.add_argument('--save_path', type=str, default='./', help='path to save')
        parser.add_argument('--prob_map', action='store_true', help='chooses outputting prob maps or binary predictions')
        parser.add_argument('--no_label', action='store_true', help='chooses if we have gt labels in testing phase')
        self.isTrain = False
        return parser
