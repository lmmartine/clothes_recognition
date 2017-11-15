function [ result ] = x_fold_CV( Data, Label, ClothesID, x, expN, classifier, para )

% all labels for x*expN experiments
Test_Label = [];
Predict_Label = [];
Prob = [];

for expi = 1:expN
    
    if strcmp(para.cv_mode, 'instance')
        
        labelNum = length(unique(Label));
        newLabel = Label;
        
        % use standard labels instead of original labels
        origLabel = sort(unique(Label));
        staLabel = Label;
        labelNum = length(origLabel);
        newLabel = 1:labelNum;
        
        
        for i=1:length(origLabel)
            staLabel(find(Label==origLabel(i)))=Inf;
            staLabel(isinf(staLabel))=newLabel(i);
        end
        Label = staLabel;
        
        %x fold CV
        for i = 1:labelNum
            categoryIndex{i} = find(Label==newLabel(i));
        end
        % store the ten fold CV index
        tenFold = cell(labelNum,x);
        
        labelNum = length(unique(Label));
        newLabel = Label;
        
        % use standard labels instead of original labels
        origLabel = sort(unique(Label));
        staLabel = Label;
        labelNum = length(origLabel);
        newLabel = 1:labelNum;
        
        
        for i=1:length(origLabel)
            staLabel(find(Label==origLabel(i)))=Inf;
            staLabel(isinf(staLabel))=newLabel(i);
        end
        Label = staLabel;
        
        %x fold CV
        for i = 1:labelNum
            categoryIndex{i} = find(Label==newLabel(i));
        end
        % store the ten fold CV index
        tenFold = cell(labelNum,x);
        
        for i = 1:labelNum
            tempIndex = categoryIndex{i};
            tempNum = length(tempIndex);
            tempLength = length(tempIndex);
            tempList = randperm(tempLength);
            tempPieceNum = fix(tempNum/x);
            tempTailNum = tempNum - x*tempPieceNum;
            for j = 1:x
                tenFold{i,j} = tempIndex(tempList(1:tempPieceNum));
                tempList(1:tempPieceNum) = [];
            end
            tempTailNum = length(tempList);
            for j = 1:tempTailNum
                tenFold{i,j} = [ tenFold{i,j};tempIndex(tempList(1)) ];
                tempList(1) = [];
            end
        end
        % the index for experiment
        experi_Index = cell(x,1);
        for i = 1:x
            temp = [];
            for j = 1:labelNum
                temp = [temp;tenFold{j,i}];
            end
            experi_Index{i} = temp;
        end
        
    elseif strcmp(para.cv_mode, 'clothes')
        
        clothes = unique(ClothesID);
        for i = 1:length(clothes)
            tmplabels = Label(find(ClothesID == clothes(i)));
            labels(i,1) = tmplabels(1);
        end
        label = unique(labels);
        labelNum = length(label);
        
        categoryIndex = cell(labelNum,1);
        %x fold CV
        for i = 1:labelNum
            categoryIndex{i} = find(labels==label(i));
        end
        % store the ten fold CV index
        tenFold = cell(labelNum,x);
        
        for i = 1:labelNum
            tempIndex = categoryIndex{i};
            tempNum = length(tempIndex);
            tempLength = length(tempIndex);
            tempList = randperm(tempLength);
            tempPieceNum = fix(tempNum/x);
            tempTailNum = tempNum - x*tempPieceNum;
            for j = 1:x
                tenFold{i,j} = tempIndex(tempList(1:tempPieceNum));
                tempList(1:tempPieceNum) = [];
            end
            tempTailNum = length(tempList);
            for j = 1:tempTailNum
                tenFold{i,j} = [ tenFold{i,j};tempIndex(tempList(1)) ];
                tempList(1) = [];
            end
        end
        % the index for experiment
        experi_Index = cell(x,1);
        for i = 1:x
            temp = [];
            for j = 1:labelNum
                temp = [temp;tenFold{j,i}];
            end
            
            for k = 1:length(temp)
                experi_Index{i} = [ experi_Index{i}; find(ClothesID==temp(k)) ];
            end
        end
        
    end
    
    for foldi = 1:x
        temp = experi_Index;
        testIndex = temp{foldi};
        temp{foldi} = [];
        trainIndex = cell2mat(temp);
        disp([num2str(foldi),'/', num2str(x),'fold CV...']);
        
        training_inst = Data(trainIndex,:);
        training_label = Label(trainIndex);
        testing_inst = Data(testIndex,:);
        testing_label = Label(testIndex);
        
        if strcmp(classifier,'SVM')
            opt = para.opt;
            model = libsvmtrain( training_label, training_inst, opt );
            [predict_label, accuracy, dec_values] = libsvmpredict( testing_label, testing_inst, model);
            prob = dec_values;
        end
        if strcmp(classifier,'adaboost')
            % options.method       = 7;
            % options.holding.rho  = 0.7;
            % options.holding.K    = 50;

            % options.weaklearner  = 0;
            % options.epsi         = 0.1;%0.1;
            % options.lambda       = 1e-2;
            % options.max_ite      = 1000000;
            % options.T            = 5;


            % model_gentle  = gentleboost_model(training_inst',training_label , options);
            % [predict_label , fxtrain]  = gentleboost_predict(testing_inst' , model_gentle);
            % prob = fxtrain;
            % % predict_label = predict_label';


            verbose = true;
            classifier = AdaBoost_mult(two_level_decision_tree, verbose); % blank classifier
            nTree = 30;
            C = classifier.train(training_inst,training_label, [], nTree);
            predict_label  = C .predict(testing_inst);
            % prob = 0;

            % verbose = true;
            % [ada_train, predict_label]= adaboost(training_inst,training_label, testing_inst)
            % classifier = AdaBoost_mult(two_level_decision_tree, verbose); % blank classifier
            % nTree = 30;
            % C = classifier.train(training_inst,training_label, [], nTree);
            % predict_label  = C .predict(testing_inst);
            prob = 0; 
        end
        if strcmp(classifier,'NN')
            options.epsilonk                      = 0.005;
            options.epsilonl                      = 0.001;
            options.epsilonlambda                 = 10e-8;
            options.sigmastart                    = 2;
            options.sigmaend                      = 10e-4;
            options.sigmastretch                  = 10e-3;
            options.threshold                     = 10e-10;
            options.xi                            = 0.1;
            options.nb_iterations                 = 3000;
            options.metric_method                 = 1;
            options.shuffle                       = 1;
            options.updatelambda                  = 1;
            options.Nproto_pclass                 = ones(1 ,  5);

            [yproto , Wproto , lambda]                       = ini_proto(training_inst' , training_label' , options.Nproto_pclass);
            [yproto_est , Wproto_est , lambda_est,  E_SRNG]  = srng_model(training_inst' , training_label' , options , yproto , Wproto , lambda);
            [predict_label , disttrain_srng]               = NN_predict(testing_inst' , yproto_est , Wproto_est , lambda_est , options);
            prob = disttrain_srng;
            predict_label = predict_label';
        end
        if strcmp(classifier,'fuzzy')
            tr_memberships = zeros(840,5);
            for k = 1 : length(training_label)
                tr_memberships(k,training_label(k)) = 1;
            end
            [y,predict_label] = f_knn(training_inst,tr_memberships,testing_inst, [1, 2 , 3, 4, 5,1,2,3,4,5]);

            for k = 1 : length(predict_label (:,1,1))
                predict_label(k,1,1) = mode(predict_label(k,1,:));
            end
            predict_label = predict_label (:,:,1);

            prob = 0; 
        end
        if strcmp(classifier,'RF')
            opt = para.opt;
            model = classRF_train( training_inst, training_label, opt.treeNum, opt.mtry );
            predict_label = classRF_predict( testing_inst, model );
            prob = 0; 
        end
        
        if strcmp(classifier, 'myGP')
            % The multi-Class Laplace Gaussian Process Classification
            
            if para.isnorm
                [ training_inst norm_obj ] = NormData(training_inst);
                testing_inst = NormData(testing_inst, norm_obj);
            end
            
            if para.model_selection
                seg = round(1/para.sampe_rate);
                tmp_training_inst = training_inst(1:seg:end,:);
                tmp_training_label = training_label(1:seg:end,:);
                [ hyp ] = modelSelection(para, tmp_training_inst, tmp_training_label);
            else
                hyp = para.hyp;
            end
            
            % estimate the posterior probility of p(f|X,Y)
            
            [ K ] = covMultiClass(hyp, para, training_inst, []);
            [ model ] = LaplaceApproximation(hyp, para, K, training_inst, training_label);
            %[ model ] = multiClassLaplaceApproximation_classic(training_inst, training_label, para);
            % predictin
            [ predict_label prob fm ] = predictGPC_classic(hyp, para, training_inst, training_label, model, testing_inst);
        end
        
        if strcmp(classifier,'GP')
            training_inst = Data(trainIndex,:);
            training_label = Label(trainIndex);
            testing_inst = Data(testIndex,:);
            testing_label = Label(testIndex);
            
            [ training_inst, norm ] = mapminmax(training_inst',0,1);training_inst = training_inst';
            training_label(training_label==1) = -1;
            training_label(training_label==2) = 1;
            
            [ testing_inst ] = mapminmax( 'apply', testing_inst' , norm ); testing_inst = testing_inst';
            testing_label(testing_label==1) = -1;
            testing_label(testing_label==2) = 1;
            
            disp('GP parameters:')
            % %             clear meanfunc covfunc likfunc hyp
            
            disp('meanfunc = @meanConst; hyp.mean = 0;')
            meanfunc = @meanConst; hyp.mean = 0;
            disp('covfunc = @covSEard;   hyp.cov = log([1 1 ... 1]);')
            covfunc = @covSEard; ell = 1.0; sf = 1.0;  hyp.cov = log([repmat(ell, 1, size(training_inst,2))], sf);
            disp('likfunc = @likErf;')
            likfunc = @likErf;
            disp('training... ')
            
            disp('hyp = minimize(hyp, @gp, -40, @infEP, meanfunc, covfunc, likfunc, x, y);')
            hyp = minimize(hyp, @gp, 1, @infEP, meanfunc, covfunc, likfunc, training_inst, training_label);
            
            disp('[a b c d lp] = gp(hyp, @infEP, meanfunc, covfunc, likfunc, x, y, t, ones(n, 1));')
            [ ymu ys2 fmu fs2 ] = gp(hyp, @infEP, meanfunc, covfunc, likfunc, training_inst, training_label, testing_inst);
            predict_label = fmu;
            predict_label(predict_label>0) = 1;
            predict_label(predict_label<=0) = -1;
        end

        

        test_label = Label(testIndex);
        Test_Label = [ Test_Label; test_label ];
        Predict_Label = [ Predict_Label; predict_label ];
        Prob = [ Prob; prob ];
        %         size(predict_label)
        % size(test_label)

        Accuracy(expi) = sum(predict_label == test_label) / length(test_label);


    end
    
end

accuracy = sum(Predict_Label == Test_Label) / length(Test_Label);
[ confMat ] = getConfusionMatrix(Test_Label, Predict_Label, para.c);

if para.flag
    figure('name','The confusion matrix');
    c = length(unique(Test_Label));
    [ Test_Label ] = label2binary(Test_Label, c, 'mat');
    [ Predict_Label ] = label2binary(Predict_Label, c, 'mat');
    plotconfusion(Test_Label', Predict_Label');
end

result.accuracy = accuracy;
result.std = std(Accuracy);
result.confMat = confMat;
result.Test_Label = Test_Label;
result.Predict_Label = Predict_Label;
result.Prob = Prob;




