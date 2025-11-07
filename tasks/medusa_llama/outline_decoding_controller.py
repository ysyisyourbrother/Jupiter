from jupiter.core.threadsafe_queue import Queue
from .kv_cache import initialize_past_key_values
class OutlineDecodingController:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(OutlineDecodingController, cls).__new__(cls)
        return cls._instance
    def __init__(self, points, config, model):
        if not hasattr(self, 'initialized'):
            self.points = points
            self.point_num = len(points)
            self.config = config
            self.model = model
            self.initialized = True 
            # points kv cache
            self.past_key_values_for_point = []
            self.past_key_values_data_for_point = []
            self.current_length_data_for_point = []
            # input_ids for points
            self.input_ids_for_point = []
            # input_len for points
            self.input_len_for_point = []
            # set up request queue
            if self.config.is_last_stage:
                self.request_queue = Queue()
            # record whether point is finish
            self.is_finish = [False]*self.point_num
            self.prepare_point_kv_cache()
    def set_up_input_ids_for_point(self, input_ids_for_point):
        self.input_ids_for_point = input_ids_for_point
        self.input_len_for_point = [ input_ids.shape[1] for input_ids in input_ids_for_point]
    def add_request(self, medusa_logits, logits, point_id):
        assert self.config.is_last_stage
        self.request_queue.add({
                        "point_id": point_id,
                        "medusa_logits": medusa_logits,
                        "logits": logits        
        })
    def add_requests(self, medusa_logits_list, logits_list):
        assert self.config.is_last_stage
        print("=====================\n init request: ", self.point_num)
        for point_id in range (self.point_num):
            self.add_request(medusa_logits_list[point_id],logits_list[point_id], point_id )
    def get_request(self ):
        assert self.config.is_last_stage
        request = self.request_queue.remove()  
        return request 
    def prepare_point_kv_cache(self):
        print("=====================\n prepare point kv cache")
        for _ in range(self.point_num):
            past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(self.model)
            self.past_key_values_for_point.append(past_key_values)
            self.past_key_values_data_for_point.append(past_key_values_data)
            self.current_length_data_for_point.append(current_length_data)

    def get_point_past_key_values_data(self,point_id):
        return self.past_key_values_data_for_point[point_id]
    def get_point_past_key_values(self,point_id ):
        return self.past_key_values_for_point[point_id]
    def get_point_current_length_data(self,point_id):
        return self.current_length_data_for_point[point_id]
    def get_input_ids(self,point_id):
        return self.input_ids_for_point[point_id]
    def update_input_ids(self,  input_ids, point_id):
        self.input_ids_for_point[point_id] = input_ids
    def get_input_len(self,point_id):
        return self.input_len_for_point[point_id]

    def get_output(self ):
        tokenizer = self.model.get_tokenizer()
        for i in range(self.point_num):
            input_len = self.input_len_for_point[i]
            input_ids = self.input_ids_for_point[i]
            text = tokenizer.decode(
                        input_ids[0, input_len :],
                        skip_special_tokens=True,
                        spaces_between_special_tokens=False,
                        clean_up_tokenization_spaces=True,
                    ) 
            print("********************\n",flush=True)
            print(self.points[i] + text,flush=True)
global controller  
def get_controller():
    return controller
def set_controller(con):
    global controller
    controller = con