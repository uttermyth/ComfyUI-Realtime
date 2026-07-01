import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { TextInputBox } from "./TextInputBox";

test("renders input with placeholder text", () => {
  render(<TextInputBox onSend={jest.fn()} />);
  expect(
    screen.getByPlaceholderText("Type a message and press Enter")
  ).toBeInTheDocument();
});

test("calls onSend with the typed text and clears the input on Send click", async () => {
  const onSend = jest.fn();
  render(<TextInputBox onSend={onSend} />);
  const input = screen.getByPlaceholderText("Type a message and press Enter");
  await userEvent.type(input, "hello");
  await userEvent.click(screen.getByRole("button", { name: /send/i }));
  expect(onSend).toHaveBeenCalledWith("hello");
  expect(input).toHaveValue("");
});

test("calls onSend when Enter is pressed", async () => {
  const onSend = jest.fn();
  render(<TextInputBox onSend={onSend} />);
  await userEvent.type(
    screen.getByPlaceholderText("Type a message and press Enter"),
    "hello{Enter}"
  );
  expect(onSend).toHaveBeenCalledWith("hello");
});
