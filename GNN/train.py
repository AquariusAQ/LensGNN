import torch
import numpy as np
import os
from datetime import datetime
import json
import time
from tqdm import tqdm
import math
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, k_hop_subgraph
from dataset_utils.dataset import get_dataset, build_subgraph, build_subgraph_new
from utils import plot_history

# SEED = 240229
# torch.manual_seed(SEED)
# np.random.seed(SEED)


def pyg_train_alternating_subgraph(model_encoder_list, model_decoder, model_encoder_name_list, data, config, device="cuda:0"):
    logs = []
    output_root = "./out/{}-{}-{}".format(config['gnn_dataset'], '-'.join(config['gnn_model']), config['gnn_output_feature_size'])
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    print("Start training...")

    epochs = config["epochs_encoders"]
    weight_decay = config["weight_decay"]
    lr_scheduler_step_size = config["lr_scheduler_step_size"]
    lr_scheduler_gamma = config["lr_scheduler_gamma"]

    t = time.perf_counter()

    print("data:", data)

    # train_size = min(len(data["train_id"]), config["batch_size"])
    train_size = min(len(data["train_id"]), config["max_train_samples"])
    val_size = min(len(data["val_id"]), config["max_val_samples"])
    test_size = min(len(data["test_id"]), config["max_test_samples"])

    train_mask = data["train_mask"].to(device)
    val_mask = data["val_mask"].to(device)
    test_mask = data["test_mask"].to(device)

    train_id = data["train_id"].numpy()
    val_id = data["val_id"].numpy()
    test_id = data["test_id"].numpy()

    data = data.to(config["subgraph_device"])

    
    ##########################
    #  train all gnn encoder #
    ##########################
    print("Start training gnn encoders...")

    optimizer_list = []
    loss_function_list = []
    scheduler_list = []

    for i in range(len(model_encoder_list)):
        learning_rate = config["learning_rate_encoders"][i]
        learning_rate_linear = config["learning_rate_linear"][i] if config["learning_rate_linear"][i] != 0 else config["learning_rate_encoders"][i] / 20
        optimizer_list.append(torch.optim.Adam([
            # {'params': model_encoder_list[i].parameters(), "lr": learning_rate, "weight_decay": weight_decay}, 
            {'params': model_encoder_list[i].convs.parameters(), "lr": learning_rate, "weight_decay": weight_decay}, 
            {'params': model_encoder_list[i].bns.parameters(), "lr": learning_rate, "weight_decay": weight_decay}, 
            {'params': model_encoder_list[i].linear.parameters(), "lr": learning_rate_linear, "weight_decay": weight_decay}, 
	        # {'params': model_decoder.parameters(), "lr": learning_rate*0.1, "weight_decay": 0.}, 
            {'params': model_decoder.parameters(), "lr": learning_rate_linear, "weight_decay": weight_decay}, 
        ]))
        loss_function_list.append(torch.nn.CrossEntropyLoss().to(device))
        scheduler_list.append(torch.optim.lr_scheduler.StepLR(optimizer_list[i], step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma))


    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "test_acc": []
    }

    
    batch_size = config["batch_size"] # batch size
    batch_per_block = 4
    block_size = batch_size * batch_per_block #

    # epoch
    for epoch in tqdm(range(epochs), desc='Epoch', leave=True):
        # torch.cuda.empty_cache()

        # 
        if epoch == config["epochs_decoder"]:
            for param in model_decoder.parameters():
                param.requires_grad = False


        ############################################
        #  #
        ############################################
        train_acc_list = [[] for i in range(len(model_encoder_list))]
        train_loss_list = [[] for i in range(len(model_encoder_list))]

        from torch_geometric.loader import DataLoader

        temp_loss_sum = 0
        temp_correct = 0

        # block
        for b in tqdm(range(math.ceil(train_size / block_size)), desc=f'Train block ({batch_per_block} batch)', leave=False):
            train_list = build_subgraph_new(data, train_id[b*block_size: min((b+1)*block_size, len(train_id))], config["layer"], device=config["subgraph_device"])
            # for g in train_list:
            #     print(g.y, g)
            train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)

            # batch
            for batch_i, batch in enumerate(train_loader):

                # encoders
                for i in range(0, len(model_encoder_list)):
                    model = model_encoder_list[i]
                    model_name = model_encoder_name_list[i]

                    # train
                    model.train()
                    model_decoder.train()

                    batch_x = batch.x  # 
                    batch_edge_index = batch.edge_index # 
                    batch_y = batch.y  # 
                    batch_batch = batch.batch  # like [0,0,1,1,1,...]

                    # torch.cuda.synchronize(config["subgraph_device"])
                    
                    batch_x = batch_x.cpu().to(config["model_device"])
                    batch_edge_index = batch_edge_index.cpu().to(config["model_device"])
                    batch_y = batch_y.cpu().to(config["model_device"])
                    batch_batch = batch_batch.cpu().to(config["model_device"])

                    # print(next(model.parameters()).device)

                    hidden = model(batch_x, batch_edge_index) 
                    # print("hidden",hidden)
                    out = model_decoder(hidden, batch_batch)
                    # print("out",out)

                    # print(out.shape, batch_y.shape)
                    # print(out)
                    # print(batch_y)

                    optimizer_list[i].zero_grad()
                    loss = loss_function_list[i](out, batch_y)
                    train_loss_list[i].append(loss.item())
                    temp_loss_sum += loss.item()

                    train_predicted = torch.argmax(out, 1)
                    train_correct = (train_predicted == batch_y).sum().item()
                    train_acc = train_correct / batch_y.shape[0]
                    train_acc_list[i].append(train_acc)
                    temp_correct += train_correct

                    loss.backward()
                    optimizer_list[i].step()

                    # print("\ttrain step loss {} acc {} ".format(loss.item(), train_acc))

                    if b == math.ceil(len(train_id) / batch_size)-1:
                        scheduler_list[i].step()
        history["train_acc"].append(temp_correct / train_size / len(model_encoder_list))
        history["train_loss"].append(temp_loss_sum / train_size * batch_size / len(model_encoder_list))


        temp_loss_sum = 0
        temp_correct = 0



        #################################
        # batch #
        #################################
        for i in range(0, len(model_encoder_list)):
            model = model_encoder_list[i]
            model_name = model_encoder_name_list[i]
            # evaluate
            model.eval()
            model_decoder.eval()

            val_loss_list = []
            val_acc_list = []

            # block
            for b in tqdm(range(math.ceil(val_size / batch_size)), desc=f'Val batch {i}', leave=False):
                val_list = build_subgraph_new(data, val_id[b*batch_size: min((b+1)*batch_size, val_size)], config["layer"], device=config["subgraph_device"])
                val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=True)

                # batch
                for batch_y_i, batch in enumerate(val_loader):

                    batch_x = batch.x.cpu().to(config["model_device"])  # 
                    batch_edge_index = batch.edge_index.cpu().to(config["model_device"])  # 
                    batch_y = batch.y.cpu().to(config["model_device"])  # 
                    batch_batch = batch.batch.cpu().to(config["model_device"])  # like [0,0,1,1,1,...]
                    

                    hidden = model(batch_x, batch_edge_index)
                    out = model_decoder(hidden, batch_batch)

                    val_loss = loss_function_list[i](out, batch_y)
                    val_loss_list.append(val_loss.item())
                    temp_loss_sum += val_loss.item()

                    val_predicted = torch.argmax(out, 1)
                    val_correct = (val_predicted == batch_y).sum().item() 
                    val_acc = val_correct / batch_y.shape[0]
                    val_acc_list.append(val_acc)
                    temp_correct += val_correct

                    # print("\tval step loss {} acc {} ".format(val_loss.item(), val_acc))

            log = '\t{}: Epoch{:4d} lr {:.6f} \ttrain_loss {:.4f} \ttrain_acc {:.2f}% \tval_loss {:.4f} \tval_acc {:.2f}%'. \
              format(model_name.upper(), epoch, optimizer_list[i].state_dict()['param_groups'][0]['lr'], 
                     sum(train_loss_list[i]) / len(train_loss_list[i]), 
                     sum(train_acc_list[i]) / len(train_acc_list[i]) * 100, 
                     sum(val_loss_list) / len(val_loss_list), 
                     sum(val_acc_list) / len(val_acc_list) * 100, )
            logs.append(log)
            if (epoch+1) % config["varbose_per_epoch"] == 0:
                print(log)
        history["val_acc"].append(temp_correct / val_size / len(model_encoder_list))
        history["val_loss"].append(temp_loss_sum / val_size * batch_size / len(model_encoder_list))


        #################################
        #  #
        #################################
        temp_correct = 0
        # test
        if epoch >= config["min_skip_epochs"] and \
            (epoch < config["epochs_decoder"] and history["val_acc"][-1] == max(history["val_acc"])) \
            or (epoch >= config["epochs_decoder"] and history["val_acc"][-1] == max(history["val_acc"][config["epochs_decoder"]:])):

            for i in range(0, len(model_encoder_list)):
                model = model_encoder_list[i]
                model_name = model_encoder_name_list[i]
                # test
                model.eval()
                model_decoder.eval()

                test_acc_list = []

                # block
                for b in tqdm(range(math.ceil(test_size / batch_size)), desc=f'Test batch {i}', leave=False):

                    test_list = build_subgraph_new(data, test_id[b*batch_size: min((b+1)*batch_size, test_size)], config["layer"], device=config["subgraph_device"])
                    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=True)

                    # batch
                    for batch_y_i, batch in enumerate(test_loader):

                        batch_x = batch.x.cpu().to(config["model_device"])  # 
                        batch_edge_index = batch.edge_index.cpu().to(config["model_device"])  # 
                        batch_y = batch.y.cpu().to(config["model_device"])  # 
                        batch_batch = batch.batch.cpu().to(config["model_device"])  # like [0,0,1,1,1,...]
                        
                        # for batch_y_i in range(batch_y.shape[0]):
                        #     batch_y[batch_y_i] = test_list[batch_size*batch_i+batch_y_i].y

                        
                        # if batch_batch[0] == batch_batch[-1]:
                        #     for ptr in range(len(batch.ptr)-1):
                        #         batch_batch[batch.ptr[ptr]:batch.ptr[ptr+1]] = ptr


                        hidden = model(batch_x, batch_edge_index)
                        out = model_decoder(hidden, batch_batch)

                        test_predicted = torch.argmax(out, 1)
                        test_correct = (test_predicted == batch_y).sum().item() 
                        test_acc = test_correct / batch_y.shape[0]
                        test_acc_list.append(test_acc)
                        temp_correct += test_correct

                        # print("\ttest step acc {} ".format(test_acc))

                log = '\t\t{}: Reach new best val_acc.\ttest_acc {:.2f}%'. \
                  format(model_name.upper(), sum(test_acc_list) / len(test_acc_list) * 100, )
                logs.append(log)
                if (epoch+1) % config["varbose_per_epoch"] == 0:
                    print(log)
            history["test_acc"].append(temp_correct / test_size / len(model_encoder_list))


        else:
            history["test_acc"].append(None)


        if epoch >= config["epochs_decoder"] and history["val_acc"][-1] == max(history["val_acc"][config["epochs_decoder"]:]) and config["save_out"] == 1:
            with open(os.path.join(output_root, "log"), "w") as f:
                # f.writelines(logs)
                f.write("\n".join(logs))
                f.close()
            with open(os.path.join(output_root, "config.json"), "w") as f:
                json.dump(config, f, indent=4)
                f.close()
            with open(os.path.join(output_root, "history.json"), "w") as f:
                json.dump(history, f, indent=4)
                f.close()
            for gnn_i, gnn_name in enumerate(config["gnn_model"]):
                torch.save(model_encoder_list[gnn_i].state_dict(), os.path.join(output_root, f"{gnn_name}.pt"))
            print("GNN encoder saved.")
            torch.save(model_decoder.state_dict(), os.path.join(output_root, f"decoder.pt"))
            print("GNN decoder saved.")

    train_time = time.perf_counter() - t
    print("Train time:", train_time)


    return


def pyg_train_alternating(model_encoder_list, model_decoder, model_encoder_name_list, data, config, device="cuda:0"):
    logs = []
    output_root = "./out/{}-{}-{}-graph".format(config['gnn_dataset'], '-'.join(config['gnn_model']), config['gnn_output_feature_size'])
    if not os.path.exists(output_root):
        os.mkdir(output_root)
    print("Start training...")

    epochs = config["epochs_encoders"]
    weight_decay = config["weight_decay"]
    lr_scheduler_step_size = config["lr_scheduler_step_size"]
    lr_scheduler_gamma = config["lr_scheduler_gamma"]

    t = time.perf_counter()

    print("data:", data)

    # train_size = min(len(data["train_id"]), config["batch_size"])
    train_size = len(data["train_id"])
    val_size = min(len(data["val_id"]), config["max_val_samples"])
    test_size = min(len(data["test_id"]), config["max_test_samples"])

    train_id = data["train_id"].numpy()
    val_id = data["val_id"].numpy()
    test_id = data["test_id"].numpy()

    x, edge_index = data["x"].to(config["model_device"]), data["edge_index"].to(config["model_device"])
    train_mask = data["train_mask"].to(config["model_device"])
    val_mask = data["val_mask"].to(config["model_device"])
    test_mask = data["test_mask"].to(config["model_device"])
    y = torch.flatten(data["y"].to(config["model_device"]))


    # data = data.to(config["subgraph_device"])

    
    ##########################
    #  train all gnn encoder #
    ##########################
    print("Start training gnn encoders...")

    optimizer_list = []
    loss_function_list = []
    scheduler_list = []

    for i in range(len(model_encoder_list)):
        learning_rate = config["learning_rate_encoders"][i]
        learning_rate_linear = config["learning_rate_linear"][i] if config["learning_rate_linear"][i] != 0 else config["learning_rate_encoders"][i] / 20
        optimizer_list.append(torch.optim.Adam([
            # {'params': model_encoder_list[i].parameters(), "lr": learning_rate, "weight_decay": weight_decay}, 
            {'params': model_encoder_list[i].convs.parameters(), "lr": learning_rate, "weight_decay": weight_decay}, 
            {'params': model_encoder_list[i].bns.parameters(), "lr": learning_rate, "weight_decay": weight_decay}, 
            {'params': model_encoder_list[i].linear.parameters(), "lr": learning_rate_linear, "weight_decay": weight_decay}, 
	        # {'params': model_decoder.parameters(), "lr": learning_rate*0.1, "weight_decay": 0.}, 
            {'params': model_decoder.parameters(), "lr": learning_rate_linear, "weight_decay": weight_decay}, 
        ]))
        loss_function_list.append(torch.nn.CrossEntropyLoss().to(device))
        scheduler_list.append(torch.optim.lr_scheduler.StepLR(optimizer_list[i], step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma))


    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "test_acc": []
    }

    best_epoch = -1
    final_best_acc = 0    
    final_best_acc_list = []
    max_val_acc = 0

    # epoch
    # for epoch in tqdm(range(epochs), desc='Epoch', leave=True):
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        if epoch == config["epochs_decoder"]:
        # if epoch >= config["epochs_decoder"]:
            for param in model_decoder.parameters():
                param.requires_grad = False
            print("Final best acc {:.2f}% (Epoch {})".format(final_best_acc*100, best_epoch))
            print(model_encoder_name_list, final_best_acc_list)
            logs.append("Final best acc {:.2f}% (Epoch {})".format(final_best_acc*100, best_epoch))
            logs.append('[' + ', '.join(model_encoder_name_list) + "] : ")
            logs.append('[' + ', '.join([str(x) for x in final_best_acc_list]) + "] : ")
            

        val_acc_list = []
        out_list = []
        hidden_list = []

        log_temp = []

        history_train_acc = 0
        history_train_loss = 0
        history_val_acc = 0
        history_val_loss = 0
        for i in range(0, len(model_encoder_list)):
            model = model_encoder_list[i]
            model_name = model_encoder_name_list[i]

            # train
            model.train()
            model_decoder.train()

            hidden = model(x, edge_index)
            out = model_decoder(hidden)

            optimizer_list[i].zero_grad()
            # print(out[train_mask].shape, y[train_mask].shape)
            # exit(0)
            loss = loss_function_list[i](out[train_mask], y[train_mask])
            train_predicted = torch.argmax(out[train_mask], 1)
            train_acc = (train_predicted == y[train_mask]).sum().item() / y[train_mask].shape[0]
            loss.backward()
            optimizer_list[i].step()
            # scheduler_list[i].step()

            history_train_acc += train_acc
            history_train_loss += loss.item()
            

            # evaluate
            model.eval()
            model_decoder.eval()

            hidden = model(x, edge_index)
            out = model_decoder(hidden)
            val_loss = loss_function_list[i](out[val_mask], y[val_mask])
            val_predicted = torch.argmax(out[val_mask], 1)
            val_acc = (val_predicted == y[val_mask]).sum().item() / y[val_mask].shape[0]

            history_val_acc += val_acc
            history_val_loss += val_loss.item()

            out_list.append(out)
            val_acc_list.append(val_acc)
            hidden_list.append(hidden)



            # print(model_name)
            log = '\t{}: Epoch{:4d} lr {:.6f} \ttrain_loss {:.4f} \ttrain_acc {:.2f}% \tval_loss {:.4f} \tval_acc {:.2f}%'. \
                  format(model_name.upper(), epoch, optimizer_list[i].state_dict()['param_groups'][0]['lr'], loss.item(), train_acc*100, val_loss, val_acc*100)
            if (epoch+1) % config["varbose_per_epoch"] == 0:
                print(log)
            logs.append(log)
            log_temp.append(log)

        history["train_loss"].append(history_train_loss / len(model_encoder_list))
        history["train_acc"].append(history_train_acc / len(model_encoder_list))
        history["val_loss"].append(history_val_loss / len(model_encoder_list))
        history["val_acc"].append(history_val_acc / len(model_encoder_list))


        val_acc_mean = sum(val_acc_list) / len(val_acc_list)
        

        # if val_acc_mean > max_val_acc and epoch + 1 > config["epochs_decoder"]:
        # test
        test_acc_list = []
        for i, out in enumerate(out_list):
            test_predicted = torch.argmax(out[test_mask], 1)
            test_acc = (test_predicted == y[test_mask]).sum().item() / y[test_mask].shape[0]
            test_acc_list.append(test_acc)
            best_epoch = epoch
        final_best_acc = sum(test_acc_list) / len(test_acc_list)
        final_best_acc_list = test_acc_list
        history["test_acc"].append(final_best_acc)
        log = 'Epoch{:4d} \taverage val_acc {:.2f}% \taverage test_acc {:.2f}%'.format(epoch, val_acc_mean*100, final_best_acc*100)
        if (epoch+1) % config["varbose_per_epoch"] != 0:
            for l in log_temp:
                print(l)
        print(log)
        logs.append(log)
            
            # out_list_cpu = []
            # hidden_list_cpu = []
        if val_acc_mean > max_val_acc and epoch + 1 > config["epochs_decoder"]:
            max_val_acc = val_acc_mean

            if history["val_acc"][-1] == max(history["val_acc"][config["epochs_decoder"]:]) and config["save_out"] == 1:
                with open(os.path.join(output_root, "log"), "w") as f:
                    # f.writelines(logs)
                    f.write("\n".join(logs))
                    f.close()
                with open(os.path.join(output_root, "config.json"), "w") as f:
                    json.dump(config, f, indent=4)
                    f.close()
                with open(os.path.join(output_root, "result.json"), "w") as f:
                    json.dump({"best_epoch": epoch,
                        "best_val_acc": history["val_acc"][-1],
                        "test_acc": test_acc_list,
                        "gnn_encoders": config["gnn_model"]
                    }, f, indent=4)
                    f.close()
                for gnn_i, gnn_name in enumerate(config["gnn_model"]):
                    torch.save(model_encoder_list[gnn_i].state_dict(), os.path.join(output_root, f"{gnn_name}.pt"))
                print("GNN encoder saved.")
                torch.save(model_decoder.state_dict(), os.path.join(output_root, f"decoder.pt"))
                print("GNN decoder saved.")
            else:
                history["test_acc"].append(None)
                log = 'Epoch{:4d} \taverage val_acc {:.2f}%'.format(epoch, val_acc_mean*100)
                if (epoch+1) % config["varbose_per_epoch"] == 0:
                    print(log)
                logs.append(log)
            
            print()
            with open(os.path.join(output_root, "history.json"), "w") as f:
                json.dump(history, f, indent=4)
                f.close()

        if (epoch+1) % 10 == 0:
            plot_history(history, output_root)


        

    train_time = time.perf_counter() - t
    print("Train time:", train_time)


    return



def train(use_models, dataset, config, enable_tokenize_cache=False, seed=0, device="cuda:0"):
    print(f"Loading dataset {dataset}...")
    data, out_feats_size, texts = get_dataset(dataset=dataset, lm=config["lm"], enable_tokenize_cache=enable_tokenize_cache, seed=seed, device=device)
    print("Dataset already load.")
    config["out_feats_size"] = out_feats_size
    text_feature_size = data.x.shape[-1]
    config["text_feature_size"] = text_feature_size

    model_encoder_list = []
    model_encoder_name_list = []
    for use_model in use_models:
        if use_model == "mlp":
            from model.MLPEncoder import PyG_MLPEncoder
            model = PyG_MLPEncoder(in_feats=text_feature_size, hidden_channels=config["hidden_channels"], out_feats=config["gnn_output_feature_size"], k=config["layer"], dropout=config["dropout"])

        elif use_model == "gcn":
            from model.GCNEncoder import PyG_GCNEncoder
            model = PyG_GCNEncoder(in_feats=text_feature_size, hidden_channels=config["hidden_channels"], out_feats=config["gnn_output_feature_size"], k=config["layer"], dropout=config["dropout"])

        elif use_model == "gat":
            from model.GATEncoder import PyG_GATEncoder
            model = PyG_GATEncoder(in_feats=text_feature_size, hidden_channels=config["hidden_channels"], out_feats=config["gnn_output_feature_size"], k=config["layer"], dropout=config["dropout"])
        
        elif use_model == "gin":
            from model.GINEncoder import PyG_GINEncoder
            model = PyG_GINEncoder(in_feats=text_feature_size, hidden_channels=config["hidden_channels"], out_feats=config["gnn_output_feature_size"], k=config["layer"], dropout=config["dropout"])
        
        elif use_model == "sage":
            from model.GraphSAGEEncoder import PyG_SAGEEncoder
            model = PyG_SAGEEncoder(in_feats=text_feature_size, hidden_channels=config["hidden_channels"], out_feats=config["gnn_output_feature_size"], k=config["layer"], dropout=config["dropout"])

        else:
            print(f"Cannot find model {use_model}")
            exit(0)
        model = model.to(config["model_device"])
        model_encoder_list.append(model)
        model_encoder_name_list.append(use_model)
        print(f"Model {use_model} structure:")
        print(model, '\n')


    print("GNN Encoder(s):", model_encoder_name_list)

    
    if config["train_type"] == "subgraph":
        from model.MLPScatterDecoder import MLPDecoder as Decoder
    elif config["train_type"] == "default":
        from model.MLPDecoder import MLPDecoder as Decoder
        # from model.ConvDecoder import ConvDecoder as Decoder
    model_decoder = Decoder(in_feats= config["gnn_output_feature_size"], out_feats=out_feats_size, hidden_channels=config["decoder_hidden_channels"], k=config["decoder_layer"])
    model_decoder = model_decoder.to(config["model_device"])

    print(model_decoder, '\n')

    # 
    if config["train_type"] == "subgraph":
        pyg_train_alternating_subgraph(model_encoder_list, model_decoder, model_encoder_name_list, data, config, device=device)
    elif config["train_type"] == "default":
        pyg_train_alternating(model_encoder_list, model_decoder, model_encoder_name_list, data, config, device=device)


import argparse
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process config.')
    
    # train config
    parser.add_argument('--train_type', type=str, default="subgraph", choices=['default','subgraph'], help='train_type')
    parser.add_argument('--varbose_per_epoch', type=int, default=1, help='varbose per epoch when training')
    parser.add_argument('--min_skip_epochs', type=int, default=0, help='min epochs when saving')
    parser.add_argument('--epochs_encoders', type=int, default=20, help='epochs_encoders')
    parser.add_argument('--epochs_decoder', type=int, default=15, help='epochs_decoder')
    parser.add_argument('--gnn_model', nargs='+', type=str, default=["gcn"], help='gnn encoder list')
    parser.add_argument('--learning_rate_encoders', nargs='+', type=float, default=[1e-2,1e-2,4e-3], help='lr for each encoder')
    parser.add_argument('--learning_rate_linear', nargs='+', type=float, default=[5e-4,5e-4,5e-4], help='lr for linear layer')
    
    
    parser.add_argument('--max_train_samples', type=int, default=512000, help='max_train_samples')
    parser.add_argument('--max_val_samples', type=int, default=512, help='max_val_samples')
    parser.add_argument('--max_test_samples', type=int, default=512, help='max_test_samples')
    parser.add_argument('--save_out', type=int, default=1, help='save_out')

    # model config
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--weight_decay', type=float, default=2e-4, help='weight_decay')
    parser.add_argument('--lr_scheduler_step_size', type=int, default=200, help='lr_scheduler_step_size')
    parser.add_argument('--lr_scheduler_gamma', type=float, default=0.95, help='lr_scheduler_gamma')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--layer', type=int, default=2, help='gnn layer')
    parser.add_argument('--hidden_channels', nargs='+', type=int, default=[256, 256], help='A list of integers')
    parser.add_argument('--gnn_output_feature_size', type=int, default=5120*8, help='gnn_output_feature_size')
    
    parser.add_argument('--decoder_layer', type=int, default=2, help='decoder_layer')
    parser.add_argument('--decoder_hidden_channels', nargs='+', type=int, default=[256], help='decoder_hidden_channels')


    # main setting
    parser.add_argument('--gnn_dataset', type=str, default="ogbn_arxiv", choices=['cora','pubmed','ogbn_arxiv','ogbn_products'], help='dataset')
    parser.add_argument('--lm', type=str, default="embed", choices=['embed', 'token'], help='lm type')
    # parser.add_argument('--save_feature', type=str, default="hidden", choices=['out', 'hidden', 'none'], help='saved feature')
    
    # cuda setting
    parser.add_argument('--model_device', type=str, default="cuda:0", help='model_device')
    parser.add_argument('--subgraph_device', type=str, default="cuda:0", help='subgraph_device')

    # abandoned ###
    parser.add_argument('--permute_encoders', action='store_true', help='permute_encoders')

    
    
    ## pubmed
    # config["learning_rate_encoders"] = [2e-4, 2e-4, 1e-4]
    ## arxiv
    # config["learning_rate_encoders"] = [1e-3, 1e-3, 5e-4]
    
    # single
    # config["epochs"] = 80
    # config["learning_rate"] = 5e-5

    args = parser.parse_args()
    print(args)
    config = vars(args)
    
    # gnn_model = ["gcn"]#, "gat", "gin"]
    gnn_dataset = config['gnn_dataset']
    gnn_model = config['gnn_model']
    
    train(gnn_model, gnn_dataset, config, enable_tokenize_cache=True, device=config["subgraph_device"])

    # abandoned
    # if config["permute_encoders"] is True:
    #     print("Attention: Traing all encoder combinations...")
    #     import itertools
    #     encoders = config["gnn_model"]
    #     for i in range(1, len(encoders)+1):
    #         combs = itertools.combinations(encoders, i)
    #         for comb in list(combs):
    #             config["gnn_model"] = list(comb)
    #             train(gnn_model, gnn_dataset, config, enable_tokenize_cache=True, device=config["subgraph_device"])
    # else:
    #     train(gnn_model, gnn_dataset, config, enable_tokenize_cache=True, device=config["subgraph_device"])


    
