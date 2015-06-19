function  omega  = kernel_matrix2(Xtrain,Kernelsextended,kernel_type, kernel_pars,Xt)

nb_data = size(Kernelsextended,1);

A=Kernelsextended;
if strcmp(kernel_type,'RBF_kernel'),
    if nargin<5,
        B=Xtrain;
    else
        B=Xt;
    end
    
    t=sum(A.^2,2)*ones(1,size(B,1));
    t2= sum(B.^2,2)*ones(1,size(A,1));
    omega=t+t2'-2*A*B';
    omega = exp(-omega./kernel_pars(1));
    
elseif strcmp(kernel_type,'lin_kernel')
    if nargin<5,
        omega = Xtrain*Xtrain';
    else
        omega = Xtrain*Xt';
    end
    
elseif strcmp(kernel_type,'poly_kernel')
    if nargin<5,
        omega = (Xtrain*Xtrain'+kernel_pars(1)).^kernel_pars(2);
    else
        omega = (Xtrain*Xt'+kernel_pars(1)).^kernel_pars(2);
    end
    
elseif strcmp(kernel_type,'wav_kernel')
    if nargin<5,
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        
        XXh1 = sum(Xtrain,2)*ones(1,nb_data);
        omega1 = XXh1-XXh1';
        omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
        
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        omega = XXh1+XXh2' - 2*(Xtrain*Xt');
        
        XXh11 = sum(Xtrain,2)*ones(1,size(Xt,1));
        XXh22 = sum(Xt,2)*ones(1,nb_data);
        omega1 = XXh11-XXh22';
        
        omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
    end
end