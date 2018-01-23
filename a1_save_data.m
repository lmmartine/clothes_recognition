function save_data( class_name, n_clothes, n_move )

name_file = ['/home/koul/bags/data/' class_name  int2str(n_clothes) '_move' int2str(n_move) '.mat'];
data_client = rossvcclient('/merge_data/get_sequence_data');
testreq = rosmessage(data_client);
testresp = call(data_client,testreq,'Timeout',10);
save(name_file,'testresp');
