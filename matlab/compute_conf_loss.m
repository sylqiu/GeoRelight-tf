function loss = compute_conf_loss(A, B ,conf)

loss = sum((A(:)-B(:)).^2) / sum(conf(:));

end