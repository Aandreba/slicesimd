use proc_macro2::{TokenStream, Ident};
use quote::{quote, format_ident, ToTokens};
use syn::{parse_macro_input, TraitItem, TraitItemConst, parse_quote, TraitItemMethod, FnArg, Receiver, PatType, TraitItemType, punctuated::Punctuated};

#[proc_macro_attribute]
pub fn simd_trait (_attrs: proc_macro::TokenStream, items: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let mut items = parse_macro_input!(items as syn::ItemTrait);
    items.supertraits.push(parse_quote! { crate::sealed::Slice });
    if items.colon_token.is_none() {
        items.colon_token = Some(Default::default());
    }

    let mut simd_items = items.clone();
    simd_items.ident = format_ident!("Simd{}", items.ident);

    let ident = &items.ident;
    let simd_ident = &simd_items.ident;
    let impls = items.items.iter().cloned().map(|item| adapt_trait_item(&simd_ident, item));

    return quote! {
        #items
        #simd_items

        impl<T: ?Sized + #simd_ident> #ident for T {
            #(#impls)*
        }
    }.into();
}

fn adapt_trait_item (name: &Ident, item: TraitItem) -> TokenStream {
    match item {
        TraitItem::Const(item) => adapt_trait_const(name, item),
        TraitItem::Method(item) => adapt_trait_method(name, item),
        TraitItem::Type(item) => adapt_trait_type(name, item),
        TraitItem::Macro(item) => item.mac.to_token_stream(),
        TraitItem::Verbatim(item) => item,
        other => syn::Error::new_spanned(other, "Unknown trait item").into_compile_error()
    }
}

fn adapt_trait_const (name: &Ident, mut item: TraitItemConst) -> TokenStream {
    let ident = &item.ident;
    item.default = Some((Default::default(), parse_quote! { <Self as #name>::#ident }));
    item.into_token_stream()
}

fn adapt_trait_method (name: &Ident, mut item: TraitItemMethod) -> TokenStream {
    let ident = &item.sig.ident;
    let inputs = item.sig.inputs.iter().map(|x| match x {
        FnArg::Receiver(Receiver { attrs, self_token, .. }) => quote! { #(#attrs)* #self_token },
        FnArg::Typed(PatType { attrs, pat, .. }) => quote! { #(#attrs)* #pat }
    });

    item.attrs.push(parse_quote! { #[inline] });
    item.default = Some(parse_quote! {{ <Self as #name>::#ident(#(#inputs),*) }});
    item.into_token_stream()
}

fn adapt_trait_type (name: &Ident, mut item: TraitItemType) -> TokenStream {
    let ident = &item.ident;
    item.colon_token = None;
    item.bounds = Punctuated::new();
    item.default = Some((Default::default(), parse_quote! { <Self as #name>::#ident }));
    item.into_token_stream()
}