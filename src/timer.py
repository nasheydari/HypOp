class Timer:
    def __init__(self) -> None:
        self.initializing_time = 0
        self.first_conv_time = 0
        self.second_conv_time = 0
        self.loss_calculate = 0
        self.loss_update = 0
        self.mapping_time = 0
        self.total_time = 0

    def update_by_timer(self, timeB):
        self.first_conv_time += timeB.first_conv_time
        self.second_conv_time += timeB.second_conv_time
        self.loss_calculate += timeB.loss_calculate
        self.loss_update += timeB.loss_update
        self.mapping_time += timeB.mapping_time
        self.total_time += timeB.multiprocessing_time

    def update_first_conv(self, time):
        self.first_conv_time = time
    
    def update_second_conv(self, time):
        self.second_conv_time = time

    def update_loss_calculate(self, time):
        self.loss_calculate = time
    
    def update_loss_update(self, time):
        self.loss_update = time

    def update_mapping_time(self, time):
        self.mapping_time = time

    def update_total_time(self, time):
        self.total_time = time
