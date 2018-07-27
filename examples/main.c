#include "darknet.h"

int main()
{
	void extract_weights(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear);
        //printf("Test ok!\n");
        char* datacfg           = "./cfg/coco.data";
        char* cfgfile           = "./cfg/yolov3-finetune-tiny.cfg";
        char* weightfile        = "../yolov3-tiny.weights";
        int *gpus; gpus = (int*)malloc(1*sizeof(int)); *gpus = 0;
        int ngpus = 1;
        int clear = 0;
        extract_weights(datacfg, cfgfile, weightfile, gpus, ngpus, clear);
        return 0;
}

void extract_weights(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    //list *options = read_data_cfg(datacfg);
    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network **nets = calloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    network *net = nets[0];
    char buff[256];
    sprintf(buff, "yolov3-finetune-tiny.weights");
    save_weights(net, buff);
    
}
